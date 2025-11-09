from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("ERROR: Couldn't open webcam. If using an image/video, set VideoCapture to that file path.")
    exit()
print("Press 'q' window to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=640)[0]  
    count = 0
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for cls in boxes.cls:
            try:
                if int(cls) == 0:
                    count += 1
            except:
                if int(float(cls)) == 0:
                    count += 1
    annotated = results.plot() 
    cv2.putText(annotated, f"Human Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Human Count (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
