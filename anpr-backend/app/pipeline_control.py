import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_PIPELINE_FILE = Path(__file__).resolve().parent / "yolo_pipeline.py"
_BACKEND_ROOT = Path(__file__).resolve().parent.parent

try:
    from app.data import cameras as _KNOWN_CAMERAS
    _KNOWN_IDS = {c["camera_id"] for c in _KNOWN_CAMERAS}
except Exception:
    _KNOWN_IDS = {"CAM01", "CAM02", "CAM03", "HYD01", "HYD02", "HYD03", "HYD04", "HYD05", "HYD06", "HYD07", "HYD08"}

_proc = None
_current_camera = None
_lock = threading.Lock()

def _is_running():
    return _proc is not None and _proc.poll() is None

@router.post("/start")
def start_pipeline(payload: dict | None = None):
    """Start app/yolo_pipeline.py for one camera, only if it is not already running."""
    global _proc, _current_camera
    camera_id = ((payload or {}).get("camera_id") or os.getenv("CAMERA_ID", "CAM01") or "CAM01").strip().upper()
    if camera_id not in _KNOWN_IDS:
        return {"running": False, "camera_id": camera_id, "error": "Unknown camera_id: " + camera_id}
    with _lock:
        if _is_running():
            return {"running": True, "already_running": True, "pid": _proc.pid, "camera_id": _current_camera}
        if not _PIPELINE_FILE.exists():
            return {
                "running": False,
                "camera_id": camera_id,
                "error": "yolo_pipeline.py not found at " + str(_PIPELINE_FILE),
            }
        env = dict(os.environ)
        env["CAMERA_ID"] = camera_id
        _proc = subprocess.Popen(
            [sys.executable, str(_PIPELINE_FILE)],
            cwd=str(_BACKEND_ROOT),
            env=env,
        )
        _current_camera = camera_id
        return {"running": True, "pid": _proc.pid, "camera_id": camera_id}

@router.post("/stop")
def stop_pipeline():
    """Stop the pipeline started by POST /api/pipeline/start."""
    global _proc, _current_camera
    with _lock:
        if not _is_running():
            _proc = None
            _current_camera = None
            return {"running": False, "already_stopped": True, "camera_id": None}
        proc = _proc
        _proc = None
        _current_camera = None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    return {"running": False, "stopped": True, "camera_id": None}

@router.get("/status")
def pipeline_status():
    """Tell the dashboard whether the pipeline is running right now, and on which camera."""
    return {"running": _is_running(), "pid": _proc.pid if _is_running() else None, "camera_id": _current_camera if _is_running() else None}
