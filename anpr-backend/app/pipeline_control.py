import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_PIPELINE_FILE = Path(__file__).resolve().parent / "yolo_pipeline.py"
_BACKEND_ROOT = Path(__file__).resolve().parent.parent

_proc = None
_lock = threading.Lock()

def _is_running():
    return _proc is not None and _proc.poll() is None

@router.post("/start")
def start_pipeline():
    """Start app/yolo_pipeline.py, only if it is not already running."""
    global _proc
    with _lock:
        if _is_running():
            return {"running": True, "already_running": True, "pid": _proc.pid}
        if not _PIPELINE_FILE.exists():
            return {
                "running": False,
                "error": "yolo_pipeline.py not found at " + str(_PIPELINE_FILE),
            }
        _proc = subprocess.Popen(
            [sys.executable, str(_PIPELINE_FILE)],
            cwd=str(_BACKEND_ROOT),
        )
        return {"running": True, "pid": _proc.pid}

@router.post("/stop")
def stop_pipeline():
    """Stop the pipeline started by POST /api/pipeline/start."""
    global _proc
    with _lock:
        if not _is_running():
            _proc = None
            return {"running": False, "already_stopped": True}
        proc = _proc
        _proc = None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    return {"running": False, "stopped": True}

@router.get("/status")
def pipeline_status():
    """Tell the dashboard whether the pipeline is running right now."""
    return {"running": _is_running(), "pid": _proc.pid if _is_running() else None}
