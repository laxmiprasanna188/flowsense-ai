# Buildathon — YOLOv8 + EasyOCR + OpenCV pipeline.
# YOLOv8 + EasyOCR + OpenCV
# - Process camera/video input.
# - Extract video frames.
# - Detect number plates using YOLOv8.
# - Recognize plate text using EasyOCR.
# - Calculate detection confidence.
# - Generate structured detection results.
# - Send detection results to your FastAPI endpoint.

import json
import os
import urllib.request

import cv2
import easyocr
from ultralytics import YOLO


def process_camera_video_input(source):
    return cv2.VideoCapture(source)


def extract_video_frames(capture):
    while True:
        success, frame = capture.read()
        if not success:
            break
        yield frame


def detect_number_plates(model, frame):
    results = model(frame)
    plates = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        plates.append((x1, y1, x2, y2, float(box.conf[0])))
    return plates


def recognize_plate_text(reader, plate_image):
    results = reader.readtext(plate_image)
    text, recognition_confidence = results[0][1], float(results[0][2])
    return text.replace(" ", "").upper(), recognition_confidence


def calculate_detection_confidence(detection_confidence, recognition_confidence):
    return (detection_confidence + recognition_confidence) / 2


def generate_structured_detection_results(camera_id, plate_text, confidence):
    return {
        "camera_id": camera_id,
        "plate_text": plate_text,
        "confidence": confidence,
    }


def send_detection_results(detection, endpoint):
    data = json.dumps(detection).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    camera_id = os.getenv("CAMERA_ID", "CAM01")
    source = os.getenv("VIDEO_SOURCE", "0")
    source = int(source) if source.isdigit() else source
    endpoint = os.getenv("FASTAPI_URL", "http://localhost:8000/api/detections")
    model = YOLO("models/yolov8-plate.pt")
    reader = easyocr.Reader(["en"])
    capture = process_camera_video_input(source)
    for frame in extract_video_frames(capture):
        for x1, y1, x2, y2, detection_confidence in detect_number_plates(model, frame):
            plate_text, recognition_confidence = recognize_plate_text(
                reader, frame[y1:y2, x1:x2]
            )
            confidence = calculate_detection_confidence(
                detection_confidence, recognition_confidence
            )
            detection = generate_structured_detection_results(
                camera_id, plate_text, confidence
            )
            send_detection_results(detection, endpoint)
    capture.release()
