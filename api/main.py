
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import videos
from api.services.report_service import AnalysisError

app = FastAPI(title="FormBack API")

# Allow communication from React during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Turn errors raised by the service layer into JSON responses.
@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# Register video routes: upload, analyze, status and results
app.include_router(
    videos.router,
    prefix="/api/videos",
    tags=["Videos"]
)


@app.get("/")
def root():
    return {"message": "FormBack API is running"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
