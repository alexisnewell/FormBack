from fastapi import APIRouter, Response

from api.schema.video import ErrorResponse, JobStatus, ResultsResponse
from api.services import analysis_service as service

router = APIRouter()


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
