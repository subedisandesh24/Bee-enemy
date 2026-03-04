import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components
import tempfile
import io
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bee & Pest Specialized Monitor", layout="wide", page_icon="🐝")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 10px; padding: 8px 20px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #ffc107 !important; color: black !important; }
</style>
""", unsafe_allow_html=True)

# --- HELPERS ---
def trigger_vibration():
    components.html("<script>if('vibrate' in navigator){navigator.vibrate([500,200,500]);}</script>", height=0, width=0)

@st.cache_resource
def load_bee_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return YOLO(os.path.join(base_dir, 'models', 'bee_best.pt'))

@st.cache_resource
def load_enemy_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return YOLO(os.path.join(base_dir, 'models', 'enemy_best.pt'))

# --- INITIALIZE ---
bee_model = load_bee_model()
enemy_model = load_enemy_model()

st.title("🐝 Hive Health: Specialized Monitoring")

tabs = st.tabs([
    "🔍 1. Bee Detector", 
    "🧬 2. Bee Species ID", 
    "🛡️ 3. Pest Detector", 
    "🦠 4. Pest Species ID", 
    "🎥 5. Video Tracking"
])

# ==========================================
# 1. BEE DETECTOR (Conf 0.30, Count Logic)
# ==========================================
with tabs[0]:
    st.header("Bee Population Scan")
    st.info("Confidence set to 0.30. Bounding boxes hidden if count ≥ 10.")
    file = st.file_uploader("Upload Hive Image", type=['jpg','png','jpeg'], key="bee_det")
    if file:
        img = Image.open(file).convert("RGB")
        results = bee_model(img, conf=0.30, verbose=False)[0]
        count = len(results.boxes)
        
        st.metric("Total Bees Detected", count)
        
        if count >= 10:
            st.warning(f"High Density: {count} bees detected. Showing clean image to avoid clutter.")
            st.image(img, use_container_width=True)
        else:
            st.success(f"Low Density: {count} bees detected. Showing bounding boxes.")
            ann_img = results.plot(line_width=2)
            st.image(ann_img, use_container_width=True)

# ==========================================
# 2. BEE SPECIES ID (Highest Conf Only)
# ==========================================
with tabs[1]:
    st.header("Top-Confidence Bee Identification")
    st.markdown("This mode analyzes the image and identifies only the **single bee** with the highest confidence score.")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="bee_class")
    if file:
        img = Image.open(file).convert("RGB")
        results = bee_model(img, conf=0.25, verbose=False)[0] # Base conf to find candidates
        
        if len(results.boxes) > 0:
            # Find the index of the highest confidence score
            conf_scores = results.boxes.conf.cpu().numpy()
            best_idx = np.argmax(conf_scores)
            
            # Filter results to only include the best detection
            best_conf = conf_scores[best_idx]
            best_class = results.names[int(results.boxes.cls[best_idx])]
            
            st.success(f"### Best Match: **{best_class}**")
            st.write(f"Inference Confidence: `{best_conf:.2f}`")
            
            # Extract only that one box and plot it
            # We filter the boxes to only the best_idx
            single_result = results[best_idx]
            st.image(single_result.plot(), caption=f"Highest Confidence Detection: {best_class}", use_container_width=True)
        else:
            st.error("No bees detected in this image to classify.")

# ==========================================
# 3. PEST DETECTOR (Conf 0.15, Count Logic)
# ==========================================
with tabs[2]:
    st.header("Bee Enemy Scanner")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_det")
    if file:
        img = Image.open(file).convert("RGB")
        results = enemy_model(img, conf=0.15, verbose=False)[0]
        count = len(results.boxes)
        
        st.metric("Pests Counted", count)
        
        if count > 5:
            st.error("🚨 ALERT: Significant Pest Activity Detected!")
            trigger_vibration()
            
        if count >= 10:
            st.info("High Pest Count. Showing raw image.")
            st.image(img, use_container_width=True)
        else:
            st.image(results.plot(line_width=2), use_container_width=True)

# ==========================================
# 4. PEST SPECIES ID (High Conf)
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_class")
    if file:
        img = Image.open(file).convert("RGB")
        # Standard high-conf classification for all detected pests
        results = enemy_model(img, conf=0.65, verbose=False)[0]
        species = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy()]
        
        if species:
            st.write("### Identified Threats (Conf 0.65+):")
            for s in set(species):
                st.write(f"- ⚠️ {s}")
            st.image(results.plot(), use_container_width=True)
        else:
            st.info("No threats identified with high confidence.")

# ==========================================
# 5. VIDEO TRACKING (Split Mode)
# ==========================================
with tabs[4]:
    st.header("Video Tracking")
    mode = st.radio("Target:", ["Bees", "Pests"], horizontal=True)
    v_file = st.file_uploader("Upload Hive Video", type=['mp4','mov','avi'])
    
    if v_file:
        selected_model = bee_model if mode == "Bees" else enemy_model
        
        # Temp file management
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t_in.write(v_file.read())
        
        cap = cv2.VideoCapture(t_in.name)
        w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        
        t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out_writer = cv2.VideoWriter(t_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        vid_placeholder = st.empty()
        unique_ids = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Using tracking function
            res = selected_model.track(frame, persist=True, conf=0.30, verbose=False)[0]
            
            if res.boxes.id is not None:
                unique_ids.update(res.boxes.id.cpu().numpy().astype(int))
            
            # Display Rule: In video, we use boxes for clarity unless count is very high
            curr_count = len(res.boxes)
            plot_frame = res.plot() if curr_count < 10 else frame
            
            # Overlay total unique count
            cv2.putText(plot_frame, f"Unique {mode} Counted: {len(unique_ids)}", (40, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            
            out_writer.write(plot_frame)
            vid_placeholder.image(cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB))
            
        cap.release()
        out_writer.release()
        
        st.success(f"Tracking Summary: Found {len(unique_ids)} unique {mode}.")
        with open(t_out.name, 'rb') as f:
            st.download_button("📥 Download Tracking Video", f, "hive_tracking.mp4")
