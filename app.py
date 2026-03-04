import os
# Fix for settings warning
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

# --- SIDEBAR ---
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
    .stImage > img { max-height: 75vh; object-fit: contain; display: block; margin: auto; }
</style>
""", unsafe_allow_html=True)

# --- HELPERS ---
def trigger_vibration():
    components.html("<script>if('vibrate' in navigator){navigator.vibrate([500,200,500]);}</script>", height=0, width=0)

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    b_path = os.path.join(base_dir, 'models', 'bee_best.pt')
    e_path = os.path.join(base_dir, 'models', 'enemy_best.pt')
    return YOLO(b_path), YOLO(e_path)

def process_image_memory_safe(file):
    """Resizes large images to prevent Streamlit Cloud OOM crashes."""
    img = Image.open(file).convert("RGB")
    # Resize if image is too large (Streamlit RAM is very limited)
    max_size = 1024
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def get_image_download(img_array):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()

# --- INITIALIZE ---
bee_model, enemy_model = load_models()

st.title("🐝 Hive Health & Security Monitor")
st.info("System optimized for stability. Confidence: 0.20")

tabs = st.tabs(["🔍 Bee Detector", "🧬 Bee Species ID", "🛡️ Pest Detector", "🦠 Pest Species ID", "🎥 Video Tracking"])

# ==========================================
# 1. BEE DETECTOR
# ==========================================
with tabs[0]:
    st.header("Bee Counter")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up1")
    if file:
        img = process_image_memory_safe(file)
        st.image(img, width=zoom_val)
        
        if st.button("🚀 Run Bee Detection", key="btn1"):
            img_cv = np.array(img)
            results = bee_model(img, conf=0.20, verbose=False)[0]
            results.names = {i: "Bee" for i in range(len(results.names))}
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            st.subheader(f"📊 {len(results.boxes)} Bees Found")
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download", get_image_download(ann_img), "bees.jpg")

# ==========================================
# 2. BEE SPECIES ID (Highest Confidence)
# ==========================================
with tabs[1]:
    st.header("Bee Species Classification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up2")
    if file:
        img = process_image_memory_safe(file)
        st.image(img, width=zoom_val)
        
        if st.button("🧬 Identify Primary Species", key="btn2"):
            results = bee_model(img, conf=0.20, verbose=False)[0]
            if len(results.boxes) > 0:
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                top = results[int(best_idx)]
                st.success(f"### {top.names[int(top.boxes.cls[0])]} (Conf: {top.boxes.conf[0]:.2f})")
                st.image(top.plot(), width=zoom_val)
            else: st.warning("No bees detected.")

# ==========================================
# 3. PEST DETECTOR
# ==========================================
with tabs[2]:
    st.header("Bee Enemy Detector")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up3")
    if file:
        img = process_image_memory_safe(file)
        st.image(img, width=zoom_val)
        
        if st.button("🛡️ Run Security Scan", key="btn3"):
            img_cv = np.array(img)
            results = enemy_model(img, conf=0.20, verbose=False)[0]
            results.names = {i: "Pest" for i in range(len(results.names))}
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            count = len(results.boxes)
            st.subheader(f"📊 {count} Pests Found")
            if count > 3:
                st.error("🚨 ALERT: Invasion Activity!")
                trigger_vibration()
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download", get_image_download(ann_img), "pests.jpg")

# ==========================================
# 4. PEST SPECIES ID (Highest Confidence)
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up4")
    if file:
        img = process_image_memory_safe(file)
        st.image(img, width=zoom_val)
        
        if st.button("🦠 Identify Primary Pest", key="btn4"):
            results = enemy_model(img, conf=0.20, verbose=False)[0]
            if len(results.boxes) > 0:
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                top = results[int(best_idx)]
                st.warning(f"### {top.names[int(top.boxes.cls[0])]} (Conf: {top.boxes.conf[0]:.2f})")
                st.image(top.plot(), width=zoom_val)
            else: st.info("No threats identified.")

# ==========================================
# 5. VIDEO TRACKING
# ==========================================
with tabs[4]:
    st.header("Video Tracking")
    mode = st.radio("Target:", ["Bees", "Pests"], horizontal=True)
    v_file = st.file_uploader("Upload Video", type=['mp4','mov','avi'])
    if v_file:
        if st.button("🎥 Start Tracking"):
            model = bee_model if mode == "Bees" else enemy_model
            t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_in.write(v_file.read())
            cap = cv2.VideoCapture(t_in.name)
            w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(t_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_holder = st.empty()
            u_ids = set()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                res = model.track(frame, persist=True, conf=0.20, verbose=False)[0]
                res.names = {i: mode[:-1] for i in range(len(res.names))}
                if res.boxes.id is not None: u_ids.update(res.boxes.id.cpu().numpy().astype(int))
                f_plot = res.plot()
                cv2.putText(f_plot, f"Total Unique: {len(u_ids)}", (40, 50), 1, 2, (255,255,255), 2)
                out.write(f_plot)
                vid_holder.image(cv2.cvtColor(f_plot, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release(); out.release()
            st.success(f"Found {len(u_ids)} unique {mode}.")
            with open(t_out.name, 'rb') as f_v: st.download_button("📥 Download", f_v, "tracking.mp4")
