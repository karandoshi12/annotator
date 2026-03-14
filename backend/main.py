"""
VLA Frame Annotation Backend
Run: uvicorn main:app --reload --port 8765
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from routes.annotate import router as annotate_router  # noqa: E402

app = FastAPI(
    title="VLA Frame Annotator",
    version="1.0.0",
    description="Annotates video frames for Vision-Language-Action robotics models",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(annotate_router, prefix="/api")

# For local development only
if __name__ == "__main__":
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    
    # Serve the frontend from the parent directory
    frontend_dir = Path(__file__).parent.parent
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=True)
