
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import uuid

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024

# Temporary in-memory storage for analysis jobs.
# Later, we will move this into a database.
ANALYSES = {}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    if Path(file.filename or "").suffix.lower() != ".mp4":
        raise HTTPException(
            status_code=400,
            detail="Only MP4 videos are supported."
        )

    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}.mp4"

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

    except Exception:
        video_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    return {
        "video_id": video_id,
        "filename": file.filename,
        "status": "uploaded",
        "size_bytes": file_size
    }


# Runs after the API responds to the analysis request.
def run_analysis(video_id: str, video_path: Path):

    try:
        from src.pipeline import analyze_video

        result = analyze_video(video_path)

        if result["error"]:
            ANALYSES[video_id] = {
                "status": "failed",
                "error": result["error"]
            }
            return

        ANALYSES[video_id] = {
            "status": "completed",
            "analysis_status": result["status"],
            "summary": result["summary"],
            "files": result["files"]
        }

    except Exception as exc:
        ANALYSES[video_id] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"
        }


@router.post("/{video_id}/analyze")
def start_analysis(
    video_id: str,
    background_tasks: BackgroundTasks
):

    video_path = UPLOAD_DIR / f"{video_id}.mp4"

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found."
        )

    existing = ANALYSES.get(video_id)

    if existing and existing["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Analysis is already processing."
        )

    ANALYSES[video_id] = {
        "status": "processing"
    }

    background_tasks.add_task(
        run_analysis,
        video_id,
        video_path
    )

    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Video analysis started."
    }


@router.get("/{video_id}/status")
def get_analysis_status(video_id: str):

    if video_id not in ANALYSES:
        raise HTTPException(
            status_code=404,
            detail="No analysis found for this video."
        )

    return {
        "video_id": video_id,
        "status": ANALYSES[video_id]["status"]
    }


@router.get("/{video_id}/results")
def get_analysis_results(video_id: str):

    if video_id not in ANALYSES:
        raise HTTPException(
            status_code=404,
            detail="No analysis found for this video."
        )

    analysis = ANALYSES[video_id]

    if analysis["status"] == "processing":
        raise HTTPException(
            status_code=202,
            detail="Analysis is still processing."
        )

    if analysis["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=analysis["error"]
        )

    return {
        "video_id": video_id,
        "status": analysis["analysis_status"],
        "summary": analysis["summary"]
    }
