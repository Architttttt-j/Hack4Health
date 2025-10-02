import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# High-priority classes (threats)
HIGH_PRIORITY = {"car", "bus", "truck", "motorbike", "bicycle",
                 "person", "dog", "cat", "stop sign", "traffic light"}

# Known real-world width (example person width ~ 40 cm, car width ~ 180 cm)
KNOWN_WIDTHS = {
    "person": 40,
    "car": 180,
    "bus": 250,
    "truck": 250,
    "motorbike": 70,
    "bicycle": 60,
    "dog": 30,
    "cat": 25
}

# Focal length (calibrated for your webcam)
FOCAL_LENGTH = 600  

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Only consider high-priority threats
            if label in HIGH_PRIORITY:
                color = (0, 0, 255)  # RED for high priority
            else:
                color = (0, 255, 0)  # Green for normal objects

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Distance estimation (only for known widths)
            if label in KNOWN_WIDTHS:
                object_width_in_pixels = x2 - x1
                if object_width_in_pixels > 0:
                    distance = (KNOWN_WIDTHS[label] * FOCAL_LENGTH) / object_width_in_pixels
                    cv2.putText(frame, f"{distance:.2f} cm",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 2)

                    # Warning alert if object is too close
                    if distance < 150:  # threshold distance in cm
                        cv2.putText(frame, "⚠ WARNING: Obstacle Ahead!", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 0, 255), 3)

    cv2.imshow("Threat Detection for Blind Assistance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
