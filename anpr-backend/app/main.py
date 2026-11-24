from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.schemas import DetectionCreate
from app.data import cameras
from app.database import SessionLocal
from app.models import Detection

app = FastAPI()


# How long a repeat sighting counts as a duplicate (seconds).
# The pipeline also dedups in memory for the same window, so a
# plate seen again within 30 seconds is stored only once.
DUPLICATE_WINDOW_SECONDS = 30

# Known cameras, from the hardcoded list in app/data.py.
# POST /api/detections used to accept ANY camera_id, even one
# that does not exist. An unknown camera is now a 400.
# An offline camera (CAM03 KIIT Square) is NOT rejected — it is
# a known camera, it just happens to be offline right now.
KNOWN_CAMERA_IDS = {camera["camera_id"] for camera in cameras}


# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Convert database record into JSON-compatible dictionary
def detection_to_dict(detection):
    return {
        "event_id": detection.event_id,
        "camera_id": detection.camera_id,
        "plate_text": detection.plate_text,
        "confidence": detection.confidence,
        "timestamp": detection.timestamp.isoformat()
        if detection.timestamp else None
    }


@app.get("/")
def home():
    return {
        "message": "Welcome to ANPR Backend!",
        "status": "Running"
    }


# POST: Save a detection in PostgreSQL
@app.post("/api/detections")
def create_detection(
    detection: DetectionCreate,
    db: Session = Depends(get_db)
):
    # Validate the camera before anything is stored. The pipeline
    # default CAM01 passes; a typo or a camera that does not exist
    # in app/data.py is a 400, not a row in the database.
    camera_id = detection.camera_id.strip()
    if camera_id not in KNOWN_CAMERA_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown camera_id: {detection.camera_id!r}. "
                f"Known cameras: {', '.join(sorted(KNOWN_CAMERA_IDS))}"
            ),
        )

    # Normalise the plate so "od02ab1234" and "OD02AB1234" dedup
    # against each other instead of being stored twice.
    plate_text = detection.plate_text.strip().upper()
    if not plate_text:
        raise HTTPException(
            status_code=400,
            detail="plate_text is required",
        )
    now = datetime.now(timezone.utc)

    # Duplicate guard: the same plate from the same camera stored
    # very recently is a repeat sighting, not a new detection.
    # Return the existing row instead of storing a duplicate.
    cutoff = now - timedelta(seconds=DUPLICATE_WINDOW_SECONDS)
    existing = (
        db.query(Detection)
        .filter(
            Detection.camera_id == camera_id,
            Detection.plate_text == plate_text,
            Detection.timestamp >= cutoff,
        )
        .order_by(Detection.timestamp.desc())
        .first()
    )
    if existing is not None:
        return {
            "message": "Duplicate detection ignored",
            "duplicate": True,
            "data": detection_to_dict(existing)
        }

    new_detection = Detection(
        event_id=str(uuid4()),
        camera_id=camera_id,
        plate_text=plate_text,
        confidence=detection.confidence,
        timestamp=now
    )

    try:
        db.add(new_detection)
        db.commit()
        db.refresh(new_detection)
    except SQLAlchemyError:
        # A failed commit must not leave the session broken for the
        # next request, and must not crash the server.
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not store detection"
        )

    return {
        "message": "Detection stored successfully",
        "data": detection_to_dict(new_detection)
    }


# GET: Retrieve all detections or filter by camera ID
@app.get("/api/detections")
def get_detections(
    camera_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(Detection)

    if camera_id:
        query = query.filter(
            Detection.camera_id == camera_id
        )

    # total_detections is the filtered total, not the page size,
    # so callers can page through all results.
    total = query.count()
    results = (
        query.order_by(Detection.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total_detections": total,
        "detections": [
            detection_to_dict(item) for item in results
        ]
    }


# GET: Retrieve one detection using event ID
@app.get("/api/detections/{event_id}")
def get_detection(
    event_id: str,
    db: Session = Depends(get_db)
):
    detection = db.query(Detection).filter(
        Detection.event_id == event_id
    ).first()

    if detection is None:
        raise HTTPException(
            status_code=404,
            detail="Detection not found"
        )

    return {
        "message": "Detection found",
        "data": detection_to_dict(detection)
    }


# GET: Retrieve all cameras
@app.get("/api/cameras")
def get_cameras():
    return {
        "total_cameras": len(cameras),
        "cameras": cameras
    }


# GET: Retrieve one camera by camera ID
@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str):

    for camera in cameras:
        if camera["camera_id"] == camera_id:
            return {
                "message": "Camera found",
                "data": camera
            }

    raise HTTPException(
        status_code=404,
        detail="Camera not found"
    )
