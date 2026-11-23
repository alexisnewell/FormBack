from pathlib import Path
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Response, UploadFile

from api.schemas.analysis import (
    AnalysisStartResponse,
    ErrorResponse,
    JobStatus,
    ReportFormat,
    ResultsResponse,
    StatusResponse,
    UploadResponse,
)
from api.services import report_service as service

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_video(file: UploadFile = File(...)):

    if Path(file.filename or "").suffix.lower() != ".mp4":
        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are supported."
        )

    video_id = str(uuid.uuid4())
    video_path = service.video_path(video_id)

    file_size = 0

    try:
        with open(video_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 500 MB limit."
                    )

                buffer.write(chunk)

        # Reject files that are not really decodable videos.
        info = service.inspect_video(video_path)

    except Exception:
        video_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    return UploadResponse(
        video_id=video_id,
        filename=file.filename,
        size_bytes=file_size,
        **info
    )


@router.post(
    "/{video_id}/analyze",
    response_model=AnalysisStartResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def start_analysis(video_id: str, background_tasks: BackgroundTasks):

    video_id = service.normalize_video_id(video_id)
    video_path = service.begin_analysis(video_id)

    background_tasks.add_task(service.run_analysis, video_id, video_path)

    return AnalysisStartResponse(
        video_id=video_id,
        status=JobStatus.processing,
        message="Video analysis started."
    )


@router.get(
    "/{video_id}/status",
    response_model=StatusResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_analysis_status(video_id: str):

    job = service.get_job(video_id)

    return StatusResponse(
        video_id=service.normalize_video_id(video_id),
        status=job.status,
        error=job.error
    )


@router.get(
    "/{video_id}/results",
    response_model=ResultsResponse,
    response_model_exclude_unset=True,
    responses={
        202: {"description": "Analysis still processing; poll again."},
        404: {"model": ErrorResponse},
    },
)
def get_analysis_results(video_id: str, response: Response):
    """
    200 with status "completed" or "failed" once the job has finished.
    202 with status "processing" while it is still running.

    A failed analysis is a finished job, not a server fault, so it is
    reported in the body (``status`` and ``error``) rather than as a 5xx.
    """

    job = service.get_job(video_id)

    if job.status == JobStatus.processing:
        response.status_code = 202

    return service.build_results(video_id, job)


@router.get(
    "/{video_id}/report",
    response_class=Response,
    responses={
        200: {
            "description": "The report file.",
            "content": {"application/pdf": {}, "text/csv": {}},
        },
        404: {"model": ErrorResponse},
        409: {
            "model": ErrorResponse,
            "description": "Still processing, failed, or low quality (no report).",
        },
    },
)
def download_report(
    video_id: str,
    fmt: ReportFormat = Query(ReportFormat.pdf, alias="format"),
):
    """Download the analysis as a PDF (with charts) or a CSV of all metrics."""

    content, media_type, filename = service.export_report(video_id, fmt)

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
