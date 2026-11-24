
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime, timezone

from app.schemas import DetectionCreate
from app.data import cameras
from app.database import SessionLocal
from app.models import Detection

app = FastAPI()


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
    new_detection = Detection(
        event_id=str(uuid4()),
        camera_id=detection.camera_id,
        plate_text=detection.plate_text,
        confidence=detection.confidence,
        timestamp=datetime.now(timezone.utc)
    )

    db.add(new_detection)
    db.commit()
    db.refresh(new_detection)

    return {
        "message": "Detection stored successfully",
        "data": detection_to_dict(new_detection)
    }


# GET: Retrieve all detections or filter by camera ID
@app.get("/api/detections")
def get_detections(
    camera_id: str | None = None,
    db: Session = Depends(get_db)
):
    query = db.query(Detection)

    if camera_id:
        query = query.filter(
            Detection.camera_id == camera_id
        )

    results = query.order_by(Detection.timestamp.desc()).all()

    return {
        "total_detections": len(results),
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
