import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import defaultdict

st.set_page_config(page_title="Human + Animal Counter", layout="wide")

# ---------- IOU ----------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# ---------- Simple Tracker ----------
class SimpleTracker:
    def __init__(self, iou_threshold=0.5, max_lost=10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self.lost = {}
        self.next_id = 0

    def update(self, detections):
        assigned = {}
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())
        iou_mat = {}
        for tid, tb in self.tracks.items():
            for didx, db in enumerate(detections):
                iou_mat[(tid, didx)] = iou(tb, db)

        for (tid, didx), score in sorted(iou_mat.items(), key=lambda x: x[1], reverse=True):
            if score < self.iou_threshold:
                continue
            if didx in unmatched_dets and tid in unmatched_tracks:
                assigned[tid] = didx
                unmatched_dets.remove(didx)
                unmatched_tracks.remove(tid)

        for tid, didx in assigned.items():
            self.tracks[tid] = detections[didx]
            self.lost[tid] = 0

        for didx in unmatched_dets:
            self.tracks[self.next_id] = detections[didx]
            self.lost[self.next_id] = 0
            self.next_id += 1

        for tid in list(unmatched_tracks):
            self.lost[tid] += 1
            if self.lost[tid] > self.max_lost:
                del self.tracks[tid]
                del self.lost[tid]

        return dict(self.tracks)

# ---------- Sidebar ----------
st.sidebar.title("Settings")
model_size = st.sidebar.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0)
conf_thres = st.sidebar.slider("Confidence threshold", 0.3, 0.9, 0.7)
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
source_type = st.sidebar.selectbox("Source", ["Upload image", "Upload video", "Webcam (0)"])
fps_limit = st.sidebar.slider("Process FPS", 1, 30, 6)
mode = st.sidebar.radio("Detection Mode", ["Humans only", "Humans + Animals"])

# ---------- Clear Button ----------
if st.sidebar.button("🧹 Clear Screen"):
    st.cache_resource.clear()
    st.experimental_rerun()

# ---------- Model Loading ----------
@st.cache_resource
def load_model(path, device):
    model = YOLO(path)
    if device == "cpu":
        model.model.to("cpu")
    return model

device = "cuda" if use_gpu else "cpu"
model = load_model(model_size, device)

# ---------- Target Classes ----------
ANIMAL_CLASSES = {"dog", "cat", "cow", "horse", "sheep", "elephant", "bird", "goat"}
if mode == "Humans only":
    TARGET_CLASSES = {"person"}
else:
    TARGET_CLASSES = {"person"} | ANIMAL_CLASSES

st.title("The Entity Plus 🧍‍♂️🐶🐄")
st.markdown("Detects and counts humans and animals using YOLOv8.")

# ---------- Streamlit placeholders ----------
placeholder = st.empty()
stats_placeholder = st.sidebar.empty()

# ---------- Uploading / Capturing ----------
if source_type == "Upload image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model.predict(frame, conf=conf_thres, verbose=False)
        r = results[0]

        dets, classes, confs = [], [], []
        if r.boxes is not None:
            for box, cls, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if name in TARGET_CLASSES and cf >= conf_thres and area > 5000:
                    dets.append([x1, y1, x2, y2])
                    classes.append(name)
                    confs.append(float(cf))

        annotated = frame.copy()
        counts = {}
        for db, cname, score in zip(dets, classes, confs):
            x1, y1, x2, y2 = db
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, f"{cname} {score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            counts[cname] = counts.get(cname, 0) + 1

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        placeholder.image(annotated_rgb, use_container_width=True)
        if counts:
            stats_placeholder.markdown("### Counts\n" + "\n".join([f"- **{k}**: {v}" for k,v in counts.items()]))
        else:
            stats_placeholder.markdown("_No detections found_")

elif source_type in ["Upload video", "Webcam (0)"]:
    if source_type == "Upload video":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        start = st.button("Start Processing")
    else:
        uploaded = None
        start = st.button("Start Webcam")

    if (uploaded and start) or (source_type == "Webcam (0)" and start):
        tracker = SimpleTracker()
        seen_ids = defaultdict(set)

        if uploaded:
            temp = uploaded.name
            with open(temp, "wb") as f:
                f.write(uploaded.getbuffer())
            cap = cv2.VideoCapture(temp)
        else:
            cap = cv2.VideoCapture(0)

        last_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            if now - last_time < 1.0 / fps_limit:
                continue
            last_time = now

            results = model.predict(frame, conf=conf_thres, verbose=False)
            r = results[0]

            dets, classes, confs = [], [], []
            if r.boxes is not None:
                for box, cls, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    name = model.names[int(cls)]
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    if name in TARGET_CLASSES and cf >= conf_thres and area > 5000:
                        dets.append([x1, y1, x2, y2])
                        classes.append(name)
                        confs.append(float(cf))

            tracks = tracker.update(dets)
            det_to_id = {}
            for tid, tb in tracks.items():
                best, best_d = 0, None
                for didx, db in enumerate(dets):
                    val = iou(tb, db)
                    if val > best:
                        best, best_d = val, didx
                if best_d is not None and best >= tracker.iou_threshold:
                    det_to_id[best_d] = tid

            annotated = frame.copy()
            for didx, db in enumerate(dets):
                x1, y1, x2, y2 = db
                cname, score = classes[didx], confs[didx]
                tid = det_to_id.get(didx)
                if tid is not None:
                    seen_ids[cname].add(tid)
                label = f"{cname} {score:.2f}" + (f" ID:{tid}" if tid is not None else "")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, label, (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            counts = {k: len(v) for k, v in seen_ids.items() if v}
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            placeholder.image(annotated_rgb, use_container_width=True)
            if counts:
                stats_placeholder.markdown("### Counts\n" + "\n".join([f"- **{k}**: {v}" for k,v in counts.items()]))
            else:
                stats_placeholder.markdown("_No detections yet_")

        cap.release()
        st.sidebar.success("Processing finished.")
