# live_entry_exit.py
import streamlit as st
from ultralytics import YOLO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Human Entry-Exit Counter", page_icon="🚶", layout="centered")
st.title("🚶 Real-Time Entry–Exit Counter (YOLOv8)")
st.write("Counts humans crossing the center line: → Entry | ← Exit")

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model


# ---------------------------
# Video Processor Class
# ---------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.entry_count = 0
        self.exit_count = 0
        self.track_memory = {}  # Track object centers across frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape

        # Define vertical center line (x = mid)
        line_x = W // 2
        cv2.line(img, (line_x, 0), (line_x, H), (0, 255, 0), 2)  # vertical line in middle

        # Run YOLO detection
        results = model(img, conf=0.4, imgsz=640, verbose=False)[0]

        # Process each person detected
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                if cls != 0:  # only person
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Draw bounding box and center point
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                # Tracking logic using object ID (if available)
                obj_id = int(box.id[0]) if box.id is not None else i

                # Store previous x positions
                prev_x = self.track_memory.get(obj_id, cx)
                self.track_memory[obj_id] = cx

                # Detect crossing
                if prev_x < line_x and cx >= line_x:
                    self.entry_count += 1
                elif prev_x > line_x and cx <= line_x:
                    self.exit_count += 1

        # Add counters on frame
        cv2.putText(img, f"Entry: {self.entry_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Exit: {self.exit_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------
# Streamlit WebRTC UI
# ---------------------------
ctx = webrtc_streamer(
    key="entry-exit-counter",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    st.markdown("###  Live Counts")
    entry_box = st.empty()
    exit_box = st.empty()

    while True:
        entry = ctx.video_processor.entry_count
        exit_ = ctx.video_processor.exit_count
        entry_box.markdown(f" **Entries:** {entry}")
        exit_box.markdown(f" **Exits:** {exit_}")

