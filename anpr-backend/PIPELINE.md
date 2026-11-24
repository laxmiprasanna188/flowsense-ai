# Buildathon — YOLOv8 + EasyOCR + OpenCV pipeline

Adds camera/video plate detection to the Buildathon FastAPI backend
(branch `feature/fastapi-postgresql-backend`). This is the group
project — separate from the solo ANPR City Tracker project, separate
cameras (Bhubaneswar, not Hyderabad), and it talks to this repo's
PostgreSQL backend, not the solo project's SQLite one.

## What it does — the 7 steps

1. **Process camera/video input** — OpenCV `VideoCapture`: a video
   file, an RTSP stream, or a webcam (`VIDEO_SOURCE=0`).
2. **Extract video frames** — reads frames, processes every
   `FRAME_SKIP`-th frame (default 5) so it keeps up on CPU.
3. **Detect number plates using YOLOv8** — a plate-trained YOLOv8
   model returns plate boxes + YOLO confidence. See the warning
   below about weights.
4. **Recognize plate text using EasyOCR** — EasyOCR reads each
   plate crop. An empty result returns no text instead of crashing.
5. **Calculate detection confidence** — simple average of the YOLO
   and OCR confidences, clamped to 0–1. Only valid plates at/above
   `MIN_CONFIDENCE` (default 0.5) are sent.
6. **Generate structured detection results** — exactly
   `{"camera_id", "plate_text", "confidence"}`.
7. **Send detection results to your FastAPI endpoint** —
   `POST /api/detections`. The backend makes up `event_id` and
   `timestamp` itself.

## Error handling

A failure in one frame or one plate never crashes the run:

- Camera/video that will not open → clear error, exit, no silent
  empty loop.
- Missing or unloadable YOLO model / EasyOCR → clear error at
  startup, exit before any frames are processed.
- Empty OCR result, empty or out-of-bounds plate crop, YOLO failure
  on one frame → that frame/plate is logged and skipped.
- API send failure → retried once, then the detection is appended
  to `failed_detections.jsonl` (with the error and a timestamp)
  instead of being lost silently, and the loop continues.
- The camera is always released (`finally`), even on Ctrl+C.
- A summary is printed at the end: frames read/processed, plates
  found, valid, sent, failed sends, skipped duplicates, skipped
  invalid, skipped low-confidence.

## Duplicate detection

Two layers, both 30 seconds:

- **Pipeline (in memory):** the same plate from the same camera
  is not re-sent within `DEDUP_SECONDS` (default 30).
- **Backend (`POST /api/detections`):** if the same `plate_text` +
  `camera_id` was already stored within the last 30 seconds, the
  existing row is returned with `"duplicate": true` and no new row
  is created. A failed database commit is rolled back and returns
  HTTP 500 instead of crashing.

## Plate validation

Only text matching the Indian plate format is sent:
2 letters + 2 digits + 1–2 letters + 4 digits, e.g. `OD02AB1234`.
A position-aware fix (e.g. `O` → `0` where a digit is required)
is only applied to 10-character text, and the result is flagged
as corrected in the log line. Anything else is skipped as invalid.

## Pagination

`GET /api/detections` now takes `limit` (default 50, max 200) and
`offset` (default 0), still newest first, still supports
`?camera_id=CAM01`. The response shape is unchanged:
`{"total_detections": <filtered total>, "detections": [...]}` —
`total_detections` is the filtered total, not the page size.

## YOLO weights — read this first

`yolov8n.pt` (COCO) has **no plate class**, so it cannot detect
plates. You need a plate-trained YOLOv8 `.pt` file (class 0 =
plate), placed at `models/yolov8-plate.pt`, and set `MODEL_PATH`
to it if it is elsewhere. **The model file is NOT in the repo
and must NOT be committed** — it is ignored by `.gitignore`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # edit DATABASE_URL / camera if needed
```

`.env.example` documents every setting: `DATABASE_URL`,
`CAMERA_ID`, `VIDEO_SOURCE`, `FASTAPI_URL`, `MODEL_PATH`,
`FRAME_SKIP`, `DEDUP_SECONDS`, `MIN_CONFIDENCE`,
`API_TIMEOUT_SECONDS`, `FAILED_LOG`.

## Run it

```bash
# 1. Start the backend first (in this repo folder)
uvicorn anpr-backend.app.main:app --reload

# 2. Run the pipeline
python anpr-backend/app/yolo_pipeline.py
```

Cameras must match the backend (`app/data.py`): CAM01 Patia
Junction (online), CAM02 Jaydev Vihar (online), CAM03 KIIT Square
(offline).

## What it does NOT do

- No vehicle tracking / trajectory, no alerts, no watchlist —
  the backend has none of those yet either.
- No auth — anyone can POST to the backend, same as before.
- No frontend / map.

## Notes

- Python 3.10 / 3.11. Do not upgrade NumPy to 2.x —
  ultralytics 8.2.51 requires numpy<2.0.0.
- EasyOCR downloads its models on first use (internet needed once).
- Syntax-checked only (`python -m py_compile`) — not run here.
  Install + weights + a running backend are needed on your
  machine first.
