import os
# Fix for Streamlit Cloud Ultralytics settings warning
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components
import tempfile
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bee & Pest Health Monitor", layout="wide", page_icon="🐝")

# --- SIDEBAR: DISPLAY CONTROLS ---
st.sidebar.header("🖼️ Display Settings")
zoom_val = st.sidebar.slider("Image Scale (Zoom)", min_value=300, max_value=2000, value=800)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 10px; padding: 8px 20px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #ffc107 !important; color: black !important; }
    .stButton>button { 
        width: 100%; border-radius: 10px; font-weight: bold; height: 3.5em; 
        background-color: #ffc107; color: black; border: 2px solid #b38600; 
    }
    .stImage > img { max-height: 80vh; object-fit: contain; display: block; margin-left: auto; margin-right: auto; }
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

def get_image_download(img_array):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()

def get_top_detection_only(results):
    """Filters YOLO results to keep only the single detection with highest confidence."""
    if not results or len(results.boxes) == 0:
        return results
    
    conf_scores = results.boxes.conf.cpu().numpy()
    best_idx = np.argmax(conf_scores)
    return results[int(best_idx)]

# --- LOAD MODELS ---
bee_model = load_bee_model()
enemy_model = load_enemy_model()

st.title("🐝 Hive Health & Security Monitor")
st.info("Confidence Level standardized to 0.20 for all detections.")

tabs = st.tabs([
    "🔍 1. Bee Detector", 
    "🧬 2. Bee Species ID", 
    "🛡️ 3. Pest Detector", 
    "🦠 4. Pest Species ID", 
    "🎥 5. Video Tracking"
])

# ==========================================
# 1. BEE DETECTOR (Generic label: Bee)
# ==========================================
with tabs[0]:
    st.header("Bee Counter")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up_bee_det")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=zoom_val, caption="Original View")
        
        if st.button("🚀 Run Bee Detection"):
            img_cv = np.array(img)
            results = bee_model(img, conf=0.20, verbose=False)[0]
            
            # Label strictly as "Bee"
            results.names = {i: "Bee" for i in range(len(results.names))}
            
            # Draw on original photo (color preserved)
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            st.subheader(f"📊 Results: {len(results.boxes)} Bees Found")
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download Annotated Image", get_image_download(ann_img), "bee_detection.jpg")

# ==========================================
# 2. BEE SPECIES ID (Highest Confidence Only)
# ==========================================
with tabs[1]:
    st.header("Bee Species Identification")
    st.markdown("Identifies only the single detection with the highest confidence.")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up_bee_sp")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=zoom_val)
        
        if st.button("🧬 Identify Primary Species"):
            results = bee_model(img, conf=0.20, verbose=False)[0]
            top_result = get_top_detection_only(results)
            
            if top_result and len(top_result.boxes) > 0:
                name = top_result.names[int(top_result.boxes.cls[0])]
                conf = top_result.boxes.conf[0]
                st.success(f"### Identified Species: {name} (Confidence: {conf:.2f})")
                ann_img = top_result.plot()
                st.image(ann_img, width=zoom_val)
                st.download_button("📥 Download ID Result", get_image_download(ann_img), "bee_id.jpg")
            else:
                st.warning("No bee identified with 0.20 confidence.")

# ==========================================
# 3. PEST DETECTOR (Generic label: Pest)
# ==========================================
with tabs[2]:
    st.header("Bee Enemy Detector")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up_pest_det")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=zoom_val)
        
        if st.button("🛡️ Run Security Scan"):
            img_cv = np.array(img)
            results = enemy_model(img, conf=0.20, verbose=False)[0]
            
            # Label strictly as "Pest"
            results.names = {i: "Pest" for i in range(len(results.names))}
            
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            count = len(results.boxes)
            st.subheader(f"📊 Results: {count} Pests Found")
            
            if count > 3:
                st.error("🚨 ALERT: Invasion Activity Detected!")
                trigger_vibration()
                
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download Annotated Image", get_image_download(ann_img), "pest_detection.jpg")

# ==========================================
# 4. PEST SPECIES ID (Highest Confidence Only)
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    st.markdown("Identifies only the single detection with the highest confidence.")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up_pest_sp")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=zoom_val)
        
        if st.button("🦠 Identify Primary Pest"):
            results = enemy_model(img, conf=0.20, verbose=False)[0]
            top_result = get_top_detection_only(results)
            
            if top_result and len(top_result.boxes) > 0:
                name = top_result.names[int(top_result.boxes.cls[0])]
                conf = top_result.boxes.conf[0]
                st.warning(f"### Detected Threat: {name} (Confidence: {conf:.2f})")
                ann_img = top_result.plot()
                st.image(ann_img, width=zoom_val)
                st.download_button("📥 Download ID Result", get_image_download(ann_img), "pest_id.jpg")
            else:
                st.info("No pests identified with 0.20 confidence.")

# ==========================================
# 5. VIDEO TRACKING
# ==========================================
with tabs[4]:
    st.header("Video Tracking Mode")
    mode = st.radio("Track Target:", ["Bees", "Pests"], horizontal=True)
    v_file = st.file_uploader("Upload Hive Video", type=['mp4','mov','avi'], key="up_video")
    
    if v_file:
        if st.button("🎥 Start Tracking Process"):
            model = bee_model if mode == "Bees" else enemy_model
            
            t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_in.write(v_file.read())
            
            cap = cv2.VideoCapture(t_in.name)
            w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            
            t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out_writer = cv2.VideoWriter(t_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            vid_holder = st.empty()
            unique_ids = set()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Run tracking
                res = model.track(frame, persist=True, conf=0.20, verbose=False)[0]
                
                # Generic label for tracking output
                res.names = {i: mode[:-1] for i in range(len(res.names))}
                
                if res.boxes.id is not None:
                    unique_ids.update(res.boxes.id.cpu().numpy().astype(int))
                
                f_plot = res.plot()
                cv2.putText(f_plot, f"Total Unique {mode}: {len(unique_ids)}", (40, 50), 1, 2, (255,255,255), 2)
                
                out_writer.write(f_plot)
                vid_holder.image(cv2.cvtColor(f_plot, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            cap.release()
            out_writer.release()
            
            st.success(f"Tracking Summary: Found {len(unique_ids)} unique {mode}.")
            with open(t_out.name, 'rb') as f_vid:
                st.download_button("📥 Download Tracking Video", f_vid, "hive_tracking.mp4")
