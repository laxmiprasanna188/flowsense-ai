# Buildathon backend — FastAPI + PostgreSQL

Stores number-plate detections from the YOLOv8 + EasyOCR pipeline
(see `PIPELINE.md`) in PostgreSQL. Eleven hardcoded cameras in two
regions — Odisha (Bhubaneswar): CAM01 Patia Junction (online),
CAM02 Jaydev Vihar (online), CAM03 KIIT Square (offline); Telangana
(Hyderabad): HYD01 Hitech City, HYD02 Gachibowli, HYD03 Kukatpally
(offline), HYD04 Banjara Hills, HYD05 Begumpet, HYD06 Secunderabad,
HYD07 Charminar, HYD08 LB Nagar (all online except HYD03).
`GET /api/cameras` returns all 11 with `city`, `region`
(`odisha` / `telangana`) and `status`.

Backend: http://127.0.0.1:8000

The dashboard is `dashboard/dashboard.html` at the repo root — one
static file, no build step. Open it and tap Connect (it tries
http://localhost:8000). Full first-time setup is in
[`../SETUP.md`](../SETUP.md).

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
| GET | `/api/cameras` | All 11 cameras (3 Odisha + 8 Hyderabad). |
| GET | `/api/cameras/{camera_id}` | One camera, or 404. |
| GET | `/api/analytics/summary` | Total detections, average confidence, per-camera counts. |
| GET | `/api/analytics/hourly` | Detections per hour (24 rows), `?start_date=&end_date=` (IST). |
| GET | `/api/analytics/daily` | Detections per day, `?start_date=&end_date=` (IST). |
| GET | `/api/analytics/cameras` | Per-camera totals and average confidence. |
| GET | `/api/vehicles/{plate_text}/trajectory` | That plate's sightings in time order, with camera names. `?limit=100` (max 500). |
| POST | `/api/alerts` | Create an alert (`alert_type`, `message`, optional `plate_text`, `camera_id`). |
| GET | `/api/alerts` | List alerts, `?status=active/acknowledged/resolved`, `?limit=50` (max 200). |
| PATCH | `/api/alerts/{alert_id}` | `{"status": "acknowledged"}` then `{"status": "resolved"}`. Resolved alerts can't change. |

No auth — anyone who can reach the backend can POST to it.

## Pipeline

`python app\yolo_pipeline.py` (backend running first, in a second
terminal). Full description, settings and the model-weights warning
are in [`PIPELINE.md`](PIPELINE.md).
