# human_count_demo.py
from ultralytics import YOLO
import cv2

# Model: yolov8n is small and fast — good for demos
model = YOLO("yolov8n.pt")  # will auto-download if not present

cap = cv2.VideoCapture(0)  # 0 = default webcam; or use "path/to/video.mp4" or "path/to/image.jpg"

if not cap.isOpened():
    print("ERROR: Couldn't open webcam. If using an image/video, set VideoCapture to that file path.")
    exit()

print("Press 'q' window to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference (results[0] is the first frame's result)
    results = model(frame, imgsz=640)[0]  

    # Count people (person class in COCO is 0)
    count = 0
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        # boxes.cls contains class indices; convert each to int and check == 0
        for cls in boxes.cls:
            try:
                if int(cls) == 0:
                    count += 1
            except:
                # fallback if object type differs
                if int(float(cls)) == 0:
                    count += 1

    # Annotated frame (YOLO provides plotting helper)
    annotated = results.plot()  # numpy image
    cv2.putText(annotated, f"Human Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Human Count (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
