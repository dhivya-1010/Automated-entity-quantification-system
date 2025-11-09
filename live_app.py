import streamlit as st
from ultralytics import YOLO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
st.set_page_config(page_title="Live Human Count", page_icon="🎥", layout="centered")
st.title("Real-Time Entity Quantification System")
st.write("AI-powered live human detection and counting using webcam feed (YOLOv8)")
model = YOLO("yolov8n.pt")  
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.4, imgsz=640)[0]
        self.count = 0
        if results.boxes is not None and len(results.boxes) > 0:
            for cls in results.boxes.cls:
                if int(cls) == 0:
                    self.count += 1
        annotated = results.plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
ctx = webrtc_streamer(
    key="human-count-stream",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
placeholder = st.empty() 
if ctx.video_processor:
    while True:
        count = ctx.video_processor.count
        placeholder.markdown(f"### Live Human Count: {count}")
