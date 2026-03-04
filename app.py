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
st.set_page_config(page_title="Bee & Pest specialized Monitor", layout="wide", page_icon="🐝")

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

# --- APP START ---
bee_model = load_bee_model()
enemy_model = load_enemy_model()

st.title("🐝 Hive Health: Specialized Monitoring")

tabs = st.tabs([
    "🔍 Bee Detector", 
    "🧬 Bee Species ID", 
    "🛡️ Pest Detector", 
    "🦠 Pest Species ID", 
    "🎥 Video Tracking"
])

# ==========================================
# 1. BEE DETECTOR (Low Conf, Count Logic)
# ==========================================
with tabs[0]:
    st.header("Bee Population Scan")
    file = st.file_uploader("Upload Hive Image", type=['jpg','png','jpeg'], key="bee_det")
    if file:
        img = Image.open(file).convert("RGB")
        results = bee_model(img, conf=0.15, verbose=False)[0]
        count = len(results.boxes)
        
        st.metric("Total Bees Detected", count)
        
        if count >= 10:
            st.info("💡 High density detected (>10). Displaying clean image for clarity.")
            st.image(img, use_container_width=True)
        else:
            st.success("💡 Low density detected (<10). Displaying bounding boxes.")
            ann_img = results.plot(line_width=2)
            st.image(ann_img, use_container_width=True)

# ==========================================
# 2. BEE SPECIES CLASSIFIER (High Conf)
# ==========================================
with tabs[1]:
    st.header("Bee Species Classification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="bee_class")
    if file:
        img = Image.open(file).convert("RGB")
        results = bee_model(img, conf=0.65, verbose=False)[0]
        species = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy()]
        
        if species:
            st.write("### Identified Species:")
            for s in set(species):
                st.write(f"- ✅ {s}")
            st.image(results.plot(), use_container_width=True)
        else:
            st.warning("No species identified with high confidence (0.65+).")

# ==========================================
# 3. PEST DETECTOR (Low Conf, Vibration)
# ==========================================
with tabs[2]:
    st.header("Bee Enemy / Pest Scanner")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_det")
    if file:
        img = Image.open(file).convert("RGB")
        results = enemy_model(img, conf=0.15, verbose=False)[0]
        count = len(results.boxes)
        
        st.metric("Pests Counted", count)
        
        if count > 5:
            st.error("🚨 ALERT: High Pest Level!")
            trigger_vibration()
            
        if count >= 10:
            st.image(img, use_container_width=True)
        else:
            st.image(results.plot(line_width=2), use_container_width=True)

# ==========================================
# 4. PEST CLASSIFIER (High Conf)
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_class")
    if file:
        img = Image.open(file).convert("RGB")
        results = enemy_model(img, conf=0.65, verbose=False)[0]
        species = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy()]
        
        if species:
            st.write("### Threat Types Detected:")
            for s in set(species):
                st.write(f"- ⚠️ {s}")
            st.image(results.plot(), use_container_width=True)
        else:
            st.info("No threats identified with high confidence.")

# ==========================================
# 5. VIDEO TRACKING (Split Mode)
# ==========================================
with tabs[4]:
    st.header("Smart Video Tracking")
    mode = st.radio("What to track?", ["Bees", "Pests (Enemies)"], horizontal=True)
    v_file = st.file_uploader("Upload Video", type=['mp4','mov','avi'])
    
    if v_file:
        model_to_use = bee_model if mode == "Bees" else enemy_model
        
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
            
            # Use track function for persistence
            res = model_to_use.track(frame, persist=True, conf=0.25, verbose=False)[0]
            
            if res.boxes.id is not None:
                unique_ids.update(res.boxes.id.cpu().numpy().astype(int))
            
            # Logic: Box display for video
            # (In video we usually want boxes, but we respect the < 10 rule if preferred)
            curr_count = len(res.boxes)
            plot_frame = res.plot() if curr_count < 10 else frame
            
            # Add counter to frame
            cv2.putText(plot_frame, f"Unique Count: {len(unique_ids)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out_writer.write(plot_frame)
            vid_placeholder.image(cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB))
            
        cap.release()
        out_writer.release()
        
        st.success(f"Finished! Total Unique {mode} identified: {len(unique_ids)}")
        with open(t_out.name, 'rb') as f:
            st.download_button("📥 Download Tracking Video", f, "processed_video.mp4")
