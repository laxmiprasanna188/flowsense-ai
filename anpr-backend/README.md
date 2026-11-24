# Buildathon backend — FastAPI + PostgreSQL

Stores number-plate detections from the YOLOv8 + EasyOCR pipeline
(see `PIPELINE.md`) in PostgreSQL. Three hardcoded cameras in
Bhubaneswar: CAM01 Patia Junction (online), CAM02 Jaydev Vihar
(online), CAM03 KIIT Square (offline).

## Setup

Run everything from this folder. Windows commands shown; on macOS /
Linux use `python3`, `source venv/bin/activate` and `cp`.

```bash
cd anpr-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env         # edit DATABASE_URL / camera if needed
python -m app.test_db          # check PostgreSQL connects (exit 0 = ok)
python -m app.create_tables    # create the detections table (once)
```

`DATABASE_URL` must name the psycopg2 driver explicitly
(`postgresql+psycopg2://...`) — plain `postgresql://` makes SQLAlchemy
2.x look for the psycopg (v3) driver, which this repo does not install.

## Run

```bash
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs.

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/` | Health check |
| POST | `/api/detections` | Store a detection (`camera_id`, `plate_text`, `confidence` 0–1). Unknown `camera_id` or empty `plate_text` is a 400. Same plate + camera within 30 s returns the existing row with `duplicate: true`. |
| GET | `/api/detections` | List detections, newest first. `?camera_id=CAM01`, `?limit=50` (max 200), `?offset=0`. `total_detections` is the filtered total. |
| GET | `/api/detections/{event_id}` | One detection, or 404. |
| GET | `/api/cameras` | All 3 cameras. |
| GET | `/api/cameras/{camera_id}` | One camera, or 404. |

No auth — anyone who can reach the backend can POST to it.

## Pipeline

`python app\yolo_pipeline.py` (backend running first, in a second
terminal). Full description, settings and the model-weights warning
are in [`PIPELINE.md`](PIPELINE.md).
