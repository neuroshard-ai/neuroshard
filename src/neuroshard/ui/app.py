"""
NeuroShard Web Dashboard

Modern web-based dashboard for monitoring and managing a NeuroShard node.
This replaces the need for a desktop GUI application.

Features:
- Real-time stats monitoring
- NEURO balance display
- Training progress tracking
- Log viewing with filters
- Resource limit controls
- Swarm architecture status
"""

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import os
import sys

# Shared state - updated by runner.py
STATE = {
    "shard_range": "Unknown",
    "peer_count": 0,
    "processed_count": 0,
    "training_updates": 0,
    "token_count": 0,
    "training_batches": 0,
    "assigned_layers": [],
    "has_embedding": False,
    "has_lm_head": False,
    "training_status": "idle",
    "throttle_cpu_ratio": 1.0,
    "throttle_ram_ratio": 1.0,
    "throttle_effective": 1.0,
}

ui_app = FastAPI(title="NeuroShard Dashboard")


def get_template_dir():
    """Get the correct template directory path."""
    # When frozen with PyInstaller, files are in sys._MEIPASS
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        template_path = os.path.join(base_path, "neuroshard", "ui", "templates")
        if os.path.exists(template_path):
            return template_path
    
    # Development mode - relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(this_dir, "templates")
    if os.path.exists(template_path):
        return template_path
    
    # Fallback
    return "neuroshard/ui/templates"


def get_static_dir():
    """Get the correct static directory path."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        static_path = os.path.join(base_path, "neuroshard", "ui", "static")
        if os.path.exists(static_path):
            return static_path
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    static_path = os.path.join(this_dir, "static")
    if os.path.exists(static_path):
        return static_path
    
    return None


templates = Jinja2Templates(directory=get_template_dir())

# Mount static files if directory exists
static_dir = get_static_dir()
if static_dir and os.path.exists(static_dir):
    ui_app.mount("/static", StaticFiles(directory=static_dir), name="static")


@ui_app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@ui_app.get("/api/stats")
async def get_dashboard_stats():
    """Get stats for the dashboard (subset of main /api/stats)."""
    return STATE


@ui_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "dashboard": "running"}
