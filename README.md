# Buildathon — ANPR backend + detection pipeline

A FastAPI + PostgreSQL backend that stores number-plate sightings,
plus a YOLOv8 + EasyOCR + OpenCV pipeline that detects plates from
a camera / video file and sends them to it.

Backend: http://127.0.0.1:8000

Cameras: 11 in two regions — Odisha (Bhubaneswar, CAM01–CAM03) and
Telangana (Hyderabad, HYD01–HYD08). 9 online, 2 offline (CAM03 KIIT
Square, HYD03 Kukatpally).

## Layout

- `anpr-backend/` — everything: FastAPI app in `app/`, pipeline in
  `app/yolo_pipeline.py`, setup and run docs in `README.md` and
  `PIPELINE.md`, dependencies in `requirements.txt`.
- `dashboard/dashboard.html` — the dashboard: one static file, no
  build step. Big Odisha / Hyderabad cards, map with 11 camera pins,
  stats, "Track any plate" (track + alerts together). Opens against
  the laptop backend at `http://localhost:8000`.
- `run.bat` — starts the backend only. `stop.bat` — stops it.
- `setup.bat` — one-click install: database, venv + install,
  `.env`, tables, and the plate model. Double-click it once.

## Quick start

First time on a new laptop? Double-click `setup.bat`, or follow
[`SETUP.md`](SETUP.md) — 7 steps, Windows / VS Code. Already set
up? Just the 2-command rerun:

```powershell
# Terminal 1 — backend (from anpr-backend)
uvicorn app.main:app --reload

# Terminal 2 — pipeline (from anpr-backend, backend running first)
python app\yolo_pipeline.py
```

Nothing starts the camera or the pipeline on its own — it runs
only when you start it in Terminal 2, or tap Start Camera in the
dashboard. Stop it with Ctrl+C there, or Stop Camera in the dashboard.

See [`anpr-backend/README.md`](anpr-backend/README.md) for the backend
and [`anpr-backend/PIPELINE.md`](anpr-backend/PIPELINE.md) for the
detection pipeline.
