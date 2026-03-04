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
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; }
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

def get_image_download(img_array, filename):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()

# --- INITIALIZE ---
bee_model = load_bee_model()
enemy_model = load_enemy_model()

st.title("🐝 Hive Health & Security Monitor")

tabs = st.tabs([
    "🔍 1. Bee Detector", 
    "🧬 2. Bee Species ID", 
    "🛡️ 3. Pest Detector", 
    "🦠 4. Pest Species ID", 
    "🎥 5. Video Tracking"
])

# ==========================================
# 1. BEE DETECTOR (Conf 0.30, Label: "Bee")
# ==========================================
with tabs[0]:
    st.header("Bee Counter")
    file = st.file_uploader("Upload Hive Image", type=['jpg','png','jpeg'], key="bee_det_file")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)
        
        if st.button("🚀 Run Bee Detection", key="btn_bee_det"):
            img_cv = np.array(img)
            results = bee_model(img, conf=0.30, verbose=False)[0]
            
            # Simplified Labeling: Override species names to just "Bee"
            results.names = {i: "Bee" for i in range(len(results.names))}
            
            # Annotate on original
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            count = len(results.boxes)
            
            st.subheader(f"Results: {count} Bees Found")
            st.image(ann_img, use_container_width=True)
            
            # Download
            st.download_button("📥 Download Annotated Image", get_image_download(ann_img, "bees.jpg"), "bee_detection.jpg", "image/jpeg")

# ==========================================
# 2. BEE SPECIES ID (Highest Conf Only)
# ==========================================
with tabs[1]:
    st.header("Bee Species Classification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="bee_class_file")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)
        
        if st.button("🧬 Identify Primary Species", key="btn_bee_class"):
            results = bee_model(img, conf=0.25, verbose=False)[0]
            
            if len(results.boxes) > 0:
                # Logic: One best detection only
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                best_box = results[best_idx]
                species_name = results.names[int(best_box.boxes.cls[0])]
                confidence = best_box.boxes.conf[0]
                
                st.success(f"### Detected: {species_name} (Conf: {confidence:.2f})")
                ann_img = best_box.plot()
                st.image(ann_img, use_container_width=True)
                st.download_button("📥 Download Result", get_image_download(ann_img, "bee_species.jpg"), "bee_species.jpg", "image/jpeg")
            else:
                st.warning("No bees detected for classification.")

# ==========================================
# 3. PEST DETECTOR (Conf 0.65, Label: "Pest")
# ==========================================
with tabs[2]:
    st.header("Bee Enemy Detector")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_det_file")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)
        
        if st.button("🛡️ Scan for Pests", key="btn_pest_det"):
            img_cv = np.array(img)
            # Conf synced to species classification (0.65)
            results = enemy_model(img, conf=0.65, verbose=False)[0]
            
            # Simplified Labeling: Override to just "Pest"
            results.names = {i: "Pest" for i in range(len(results.names))}
            
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            count = len(results.boxes)
            
            st.subheader(f"Results: {count} Pests Found")
            if count > 5:
                st.error("🚨 HIGH PEST ACTIVITY DETECTED!")
                trigger_vibration()
                
            st.image(ann_img, use_container_width=True)
            st.download_button("📥 Download Annotated Image", get_image_download(ann_img, "pests.jpg"), "pest_detection.jpg", "image/jpeg")

# ==========================================
# 4. PEST SPECIES ID (Conf 0.65)
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_class_file")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)
        
        if st.button("🦠 Identify Pest Species", key="btn_pest_class"):
            results = enemy_model(img, conf=0.65, verbose=False)[0]
            species_found = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy()]
            
            if species_found:
                st.write("### Threat Analysis:")
                for s in set(species_found):
                    st.write(f"- ⚠️ {s}")
                ann_img = results.plot()
                st.image(ann_img, use_container_width=True)
                st.download_button("📥 Download Result", get_image_download(ann_img, "pest_species.jpg"), "pest_species.jpg", "image/jpeg")
            else:
                st.info("No pests detected with high confidence (0.65).")

# ==========================================
# 5. VIDEO TRACKING (Bees / Pests)
# ==========================================
with tabs[4]:
    st.header("Video Tracking Mode")
    mode = st.radio("Track Target:", ["Bees", "Pests"], horizontal=True)
    v_file = st.file_uploader("Upload Video File", type=['mp4','mov','avi'], key="video_file")
    
    if v_file:
        if st.button(f"🎥 Start Tracking {mode}", key="btn_video"):
            model = bee_model if mode == "Bees" else enemy_model
            conf_val = 0.30 if mode == "Bees" else 0.65
            
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
                
                # Tracking logic
                res = model.track(frame, persist=True, conf=conf_val, verbose=False)[0]
                
                # Simple labels for video output
                res.names = {i: mode[:-1] for i in range(len(res.names))} 
                
                if res.boxes.id is not None:
                    unique_ids.update(res.boxes.id.cpu().numpy().astype(int))
                
                plot_frame = res.plot()
                cv2.putText(plot_frame, f"Unique {mode}: {len(unique_ids)}", (40, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out_writer.write(plot_frame)
                vid_placeholder.image(cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB))
                
            cap.release()
            out_writer.release()
            
            st.success(f"Analysis Complete: {len(unique_ids)} unique {mode} tracked.")
            with open(t_out.name, 'rb') as f:
                st.download_button("📥 Download Tracked Video", f, "hive_tracking.mp4")
