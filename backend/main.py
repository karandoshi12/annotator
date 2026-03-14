"""
VLA Frame Annotation Backend
Run: python3 main.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

# API routes must come BEFORE static files
app.include_router(annotate_router, prefix="/api")

# Serve static files (frontend) - this should be last
frontend_dir = Path(__file__).parent.parent
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")

# For local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=True)
