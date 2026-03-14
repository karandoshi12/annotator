"""
Vercel serverless function entry point
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from main import app

# This is required for Vercel
handler = app
