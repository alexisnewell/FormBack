
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import videos

app = FastAPI(title="FormBack API")

# Allow communication from React during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register video routes
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
