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

# --- SIDEBAR CONTROL (Zoom Function) ---
st.sidebar.header("🖼️ Display Settings")
zoom_level = st.sidebar.slider("Image Display Scale (%)", min_value=30, max_value=100, value=80)
display_width = int(zoom_level * 10) # Translates slider to pixel-relative width

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6; border-radius: 10px; padding: 8px 20px; font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{ background-color: #ffc107 !important; color: black !important; }}
    .stButton>button {{ width: 100%; border-radius: 10px; font-weight: bold; height: 3em; background-color: #ffc107; color: black; }}
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

def smart_filter_pests(results):
    """
    Logic: Apply high confidence to Hornets to prevent Bee confusion,
    and moderate confidence to other pests.
    """
    boxes = results.boxes
    keep_indices = []
    
    for i, (cls_idx, conf) in enumerate(zip(boxes.cls, boxes.conf)):
        class_name = results.names[int(cls_idx)].lower()
        
        # Hornet threshold is high (0.65) to avoid bee confusion
        if "hornet" in class_name or "wasp" in class_name:
            if conf >= 0.65:
                keep_indices.append(i)
        # Other pests use lower threshold (0.35)
        else:
            if conf >= 0.35:
                keep_indices.append(i)
                
    return results[keep_indices]

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
# 1. BEE DETECTOR
# ==========================================
with tabs[0]:
    st.header("Bee Counter")
    file = st.file_uploader("Upload Hive Image", type=['jpg','png','jpeg'], key="bee_det_up")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=display_width, caption="Original View")
        
        if st.button("🚀 Run Bee Detection", key="btn_bee_det"):
            img_cv = np.array(img)
            results = bee_model(img, conf=0.30, verbose=False)[0]
            
            # Simple label "Bee"
            results.names = {i: "Bee" for i in range(len(results.names))}
            
            ann_img = results.plot(img=img_cv.copy(), line_width=2)
            st.subheader(f"📊 {len(results.boxes)} Bees Identified")
            st.image(ann_img, width=display_width)
            st.download_button("📥 Download Photo", get_image_download(ann_img), "bee_count.jpg")

# ==========================================
# 2. BEE SPECIES ID (Top Match)
# ==========================================
with tabs[1]:
    st.header("Bee Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="bee_spec_up")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=display_width)
        
        if st.button("🧬 Identify Species", key="btn_bee_spec"):
            results = bee_model(img, conf=0.25, verbose=False)[0]
            if len(results.boxes) > 0:
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                res = results[best_idx]
                st.success(f"### Result: {results.names[int(res.boxes.cls[0])]} ({res.boxes.conf[0]:.2f})")
                ann_img = res.plot()
                st.image(ann_img, width=display_width)
            else:
                st.warning("No bee species identified.")

# ==========================================
# 3. PEST DETECTOR (Smart Variable Conf)
# ==========================================
with tabs[2]:
    st.header("Pest Detection (Bee Enemy)")
    st.info("💡 Variable Threshold: Hornets (0.65) | Others (0.35) to prevent confusion with bees.")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_det_up")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=display_width)
        
        if st.button("🛡️ Run Security Scan", key="btn_pest_det"):
            img_cv = np.array(img)
            # Run with baseline low conf, then filter in python
            raw_results = enemy_model(img, conf=0.20, verbose=False)[0]
            filtered_results = smart_filter_pests(raw_results)
            
            # Label as "Pest" only
            filtered_results.names = {i: "Pest" for i in range(len(filtered_results.names))}
            
            ann_img = filtered_results.plot(img=img_cv.copy(), line_width=2)
            count = len(filtered_results.boxes)
            
            st.subheader(f"📊 {count} Pests Detected")
            if count > 3:
                st.error("🚨 ALERT: Invasion Risk!")
                trigger_vibration()
                
            st.image(ann_img, width=display_width)
            st.download_button("📥 Download Photo", get_image_download(ann_img), "pest_count.jpg")

# ==========================================
# 4. PEST SPECIES ID (Variable Conf)
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="pest_spec_up")
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=display_width)
        
        if st.button("🦠 ID Pest Species", key="btn_pest_spec"):
            raw_results = enemy_model(img, conf=0.20, verbose=False)[0]
            filtered = smart_filter_pests(raw_results)
            
            if len(filtered.boxes) > 0:
                species = [filtered.names[int(c)] for c in filtered.boxes.cls.cpu().numpy()]
                st.write("### Species Found:")
                for s in set(species):
                    st.write(f"- ⚠️ {s}")
                ann_img = filtered.plot()
                st.image(ann_img, width=display_width)
            else:
                st.info("No verified threats found.")

# ==========================================
# 5. VIDEO TRACKING
# ==========================================
with tabs[4]:
    st.header("Smart Tracking")
    mode = st.radio("Track:", ["Bees", "Pests"], horizontal=True)
    v_file = st.file_uploader("Upload Video", type=['mp4','mov','avi'], key="vid_up")
    
    if v_file:
        if st.button("🎥 Start Tracking", key="btn_track"):
            model = bee_model if mode == "Bees" else enemy_model
            
            t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_in.write(v_file.read())
            
            cap = cv2.VideoCapture(t_in.name)
            w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(t_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            vid_holder = st.empty()
            unique_ids = set()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Apply variable conf logic for pests in video too
                if mode == "Pests":
                    res_raw = model.track(frame, persist=True, conf=0.25, verbose=False)[0]
                    res = smart_filter_pests(res_raw)
                else:
                    res = model.track(frame, persist=True, conf=0.30, verbose=False)[0]
                
                res.names = {i: mode[:-1] for i in range(len(res.names))}
                if res.boxes.id is not None:
                    unique_ids.update(res.boxes.id.cpu().numpy().astype(int))
                
                f_plot = res.plot()
                cv2.putText(f_plot, f"Total Unique: {len(unique_ids)}", (30, 40), 1, 2, (255,255,255), 2)
                
                out.write(f_plot)
                vid_holder.image(cv2.cvtColor(f_plot, cv2.COLOR_BGR2RGB), width=display_width)
                
            cap.release()
            out.release()
            st.success(f"Tracked {len(unique_ids)} unique {mode}.")
            with open(t_out.name, 'rb') as f:
                st.download_button("📥 Download Video", f, "tracking.mp4")
