import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    results = model(frame, conf=0.25)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Snack Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()