Buildathon - pipeline control
==========================================

Where each file in this folder goes (project folder = D:\flowsense-ai):
  dashboard.html        -> D:\flowsense-ai\dashboard\dashboard.html
                           (replaces the old dashboard.html)
  run.bat               -> D:\flowsense-ai\run.bat
  stop.bat              -> D:\flowsense-ai\stop.bat
  pipeline_control.py   -> D:\flowsense-ai\anpr-backend\app\pipeline_control.py
  this file             -> read it, no need to copy it

1. Pipeline control is already in the backend
------------------------------------------
main.py already includes pipeline_control.py, so no change
to main.py is needed.

2. Restart the backend
----------------------
Double-click stop.bat (or close the black Backend window),
then double-click run.bat. run.bat starts ONLY the backend.
The pipeline does NOT start with run.bat any more.

3. How Start / Stop Camera works now
------------------------------------
  Start Camera in dashboard.html does two things:
    (a) starts the <video> preview with getUserMedia, and
    (b) calls POST http://localhost:8000/api/pipeline/start,
        which starts "python app/yolo_pipeline.py" on your laptop.
    Status then says: Camera + detection running

  Stop Camera does two things:
    (a) stops all video tracks (preview off), and
    (b) calls POST http://localhost:8000/api/pipeline/stop.
    Status then says: Camera + detection stopped

  GET http://localhost:8000/api/pipeline/status tells the
  dashboard whether the pipeline is running right now.

  If the backend does not have the control endpoint yet
  (you have not done step 1 above), Start Camera shows:
    Add pipeline_control.py to the backend first — see PIPELINE_CONTROL_README.txt
  It does not pretend detection started.

4. One camera, two users
------------------------
The laptop camera can be used by one app at a time, so the
preview and the pipeline share the same camera — the dashboard
preview is for framing, detection uses the same feed via the
pipeline. If the preview will not open while detection is
running, that is the camera being busy. Detection is still
running. Tap Stop Camera to free the camera again.

5. If the pipeline will not start
---------------------------------
  - Check the black Backend terminal window for the error.
  - Check models\yolov8-plate.pt exists in anpr-backend\models.
  - Check the venv has cv2, easyocr and ultralytics installed.
