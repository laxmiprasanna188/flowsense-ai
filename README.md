# Buildathon — ANPR backend + detection pipeline

Group hackathon project: a FastAPI + PostgreSQL backend that stores
number-plate sightings, plus a YOLOv8 + EasyOCR + OpenCV pipeline
that detects plates from a camera / video file and sends them to it.

Cameras are in Bhubaneswar: CAM01 Patia Junction, CAM02 Jaydev Vihar,
CAM03 KIIT Square. This is the group project — separate from the solo
ANPR City Tracker project.

## Layout

- `anpr-backend/` — everything: FastAPI app in `app/`, pipeline in
  `app/yolo_pipeline.py`, setup and run docs in `README.md` and
  `PIPELINE.md`, dependencies in `requirements.txt`.

## Quick start

See [`anpr-backend/README.md`](anpr-backend/README.md) for the backend
and [`anpr-backend/PIPELINE.md`](anpr-backend/PIPELINE.md) for the
detection pipeline.
