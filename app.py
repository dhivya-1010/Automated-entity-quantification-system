# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np

st.set_page_config(page_title="Human Count System", page_icon="🧍", layout="centered")
st.title("🧍 Human Count System")
st.write("Upload an image or video and the AI will count how many humans are present!")

model = YOLO("yolov8n.pt")

choice = st.radio("Choose Input Type", ["Image", "Video"])

if choice == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        img_array = np.array(img)
        results = model(img_array)[0]
        count = sum(1 for c in results.boxes.cls if int(c) == 0)

        st.image(results.plot(), caption=f"Detected Humans: {count}", use_column_width=True)
        st.success(f"✅ Human Count: {count}")

elif choice == "Video":
    file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        vid_cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if not ret:
                break
            results = model(frame)[0]
            count = sum(1 for c in results.boxes.cls if int(c) == 0)
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated, caption=f"Humans: {count}", use_column_width=True)

        vid_cap.release()
        st.success("✅ Video Processing Complete!")
