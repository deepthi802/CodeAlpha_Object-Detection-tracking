import cv2
from ultralytics import YOLO
from sort import *

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

# Open webcam
cap = cv2.VideoCapture(0)

# For video file use:
# cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
