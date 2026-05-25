import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Webcam or video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            if conf > 0.5:
                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], conf, label)
                )

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        label = track.get_det_class()

        # Draw box
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      2)

        # Draw label
        cv2.putText(frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    # Show output
    cv2.imshow("Object Detection and Tracking", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
