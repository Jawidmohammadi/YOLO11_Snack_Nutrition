import cv2
import yaml
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

with open("nutrition_data.yaml", "r") as f:
    nutrition_config = yaml.safe_load(f)

nutrition_map = {}
for item in nutrition_config["classes"]:
    name = item["name"]
    nutrition_map[name] = item["nutrition"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    results = model(frame, conf=0.25, verbose=False)
    detections = results[0].boxes
    labels = model.names

    total_calories = 0
    total_protein = 0
    total_sugar = 0
    total_carb = 0
    object_count = 0

    for box in detections:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = labels[class_id]

        if conf < 0.25:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        object_count += 1

        calories = nutrition_map.get(class_name, {}).get("calories_per_serving", 0)
        protein = nutrition_map.get(class_name, {}).get("protein_grams", 0)
        sugar = nutrition_map.get(class_name, {}).get("sugar_grams", 0)
        carb = nutrition_map.get(class_name, {}).get("carb_grams", 0)

        total_calories += calories
        total_protein += protein
        total_sugar += sugar
        total_carb += carb

        label = f"{class_name} {conf:.2f} | {calories} cal"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )



    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (420, 170), (0, 0, 0), -1)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"Objects: {object_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"Total Calories: {total_calories}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"Total Protein: {total_protein}g", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"Total Sugar: {total_sugar}g", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"Total Carbohydrate: {total_carb}g", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("YOLO Snack Nutrition Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()
