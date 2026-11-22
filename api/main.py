
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import videos
from api.services.report_service import AnalysisError


app = FastAPI(title="FormBack API")


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"


# Allow React development server to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Handle analysis errors
@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# Register API routes before the frontend
app.include_router(
    videos.router,
    prefix="/api/videos",
    tags=["Videos"],
)


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


# Serve the React frontend
@app.get("/", include_in_schema=False)
def serve_frontend():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return JSONResponse(
        status_code=503,
        content={
            "message": "FormBack API is running",
            "detail": "Frontend not built. Run npm install and npm run build inside frontend.",
        },
    )


# Serve React assets (JS, CSS, images, etc.)
if FRONTEND_DIST.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="frontend",
    )
