import os
# Fix for Streamlit Cloud settings warning
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

# --- SIDEBAR DISPLAY ---
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
    .footer { text-align: center; padding: 20px; font-weight: bold; color: #5a4609; border-top: 1px solid #ddd; margin-top: 50px;}
    .bee-info { background-color: #fff9e6; padding: 20px; border-radius: 15px; border-left: 5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# --- BEE SPECIES DATA ---
BEE_PROFILES = {
    "Apis laboriosa": """
        <div class="bee-info">
        <h3>🍯 <b>Apis laboriosa</b></h3>
        <p><i>Commonly called: <b>Himalayan Giant Honey Bee / Bhir Mauri</b></i></p>
        <ul>
            <li><b>Identification & Nesting:</b> The world's largest honeybee! They feature a dark abdomen specifically adapted to absorb heat in freezing mountain climates. They build massive, single, open-air combs under the overhangs of sheer cliffs.</li>
            <li><b>Location in Nepal:</b> High-altitude Himalayan belts (1,200 to 3,600 masl). Found extensively in the cliffs of <b>Kaski, Lamjung, Myagdi, Manaslu, and Solukhumbu</b>.</li>
            <li><b>Botanical Foraging:</b> They feed heavily on nectar from the <b>Rhododendron (Lali Gurans)</b> family.</li>
            <li><b>Nepalese Context:</b> Known for producing the famous <b>"Mad Honey" (Red Honey)</b> with psychoactive properties. They are the focal point of ancient Gurung honey-hunting traditions.</li>
        </ul>
        </div>
    """,
    "Apis dorsata": """
        <div class="bee-info">
        <h3>🌳 <b>Apis dorsata</b></h3>
        <p><i>Commonly called: <b>Giant Honey Bee / Khag Mauri / Kadam Mauri</b></i></p>
        <ul>
            <li><b>Identification & Nesting:</b> A large, aggressive bee with a distinct yellow/black banded abdomen. They build large combs on tall trees, building overhangs, and water towers.</li>
            <li><b>Location in Nepal:</b> Terai plains and lower Churia hills up to 1,200 masl. They are <b>highly migratory</b>.</li>
            <li><b>Botanical Foraging:</b> Vital for <b>Sal forests (Shorea robusta)</b> and agricultural fields like <b>Mustard (Tori)</b>.</li>
            <li><b>Significance:</b> The backbone of natural pollination in the Terai region; traditionally hunted by the <b>Tharu community</b>.</li>
        </ul>
        </div>
    """,
    "Apis cerana": """
        <div class="bee-info">
        <h3>🏠 <b>Apis cerana</b></h3>
        <p><i>Commonly called: <b>Asian Hive Bee / Ghar Mauri</b></i></p>
        <ul>
            <li><b>Identification & Nesting:</b> Medium-sized and docile. They are <b>cavity nesters</b>, making them perfect for traditional domestic wall hives (Khopa).</li>
            <li><b>Location in Nepal:</b> Widely distributed in mid-hills and mountains (600 to 3,000+ masl) from <b>Jumla to Ilam</b>.</li>
            <li><b>Botanical Foraging:</b> Highly dependent on <b>Chiuri (Indian Butter Tree)</b>, producing distinct white honey.</li>
            <li><b>Significance:</b> Naturally resistant to local mites and perfectly adapted to Nepal's harsh winters.</li>
        </ul>
        </div>
    """,
    "Apis florea": """
        <div class="bee-info">
        <h3>🦋 <b>Apis florea</b></h3>
        <p><i>Commonly called: <b>Little Honey Bee / Kathya Mauri / Putali Mauri</b></i></p>
        <ul>
            <li><b>Identification & Nesting:</b> The smallest species. They build single, palm-sized combs wrapped around branches in dense shrubs.</li>
            <li><b>Location in Nepal:</b> Subtropical regions like the <b>Terai and hot river valleys</b>.</li>
            <li><b>Botanical Foraging:</b> Heroes of <b>micro-pollination</b>, visiting small wild weeds and fruit trees like Mango.</li>
            <li><b>Significance:</b> Essential for maintaining biodiversity in lowland shrubs that larger bees ignore.</li>
        </ul>
        </div>
    """,
    "Apis mellifera": """
        <div class="bee-info">
        <h3>📦 <b>Apis mellifera</b></h3>
        <p><i>Commonly called: <b>European Honey Bee / Bideshi Mauri</b></i></p>
        <ul>
            <li><b>Identification & Nesting:</b> Golden-brown coloration. Introduced species kept in modern <b>Langstroth hives</b>.</li>
            <li><b>Location in Nepal:</b> Commercial hubs in <b>Chitwan, Dang, and Nawalparasi</b>.</li>
            <li><b>Botanical Foraging:</b> Managed alongside massive commercial blooms like <b>Winter Mustard</b> and Litchi.</li>
            <li><b>Significance:</b> Used for high-yield commercial honey production, but very vulnerable to the <b>Asian Giant Hornet</b>.</li>
        </ul>
        </div>
    """
}

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
    img = Image.open(file).convert("RGB")
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

tabs = st.tabs(["🔍 Bee Detector", "🧬 Bee Species ID", "🛡️ Pest Detector", "🦠 Pest Species ID", "🎥 Video Tracking"])

# ==========================================
# 1. BEE DETECTOR (Adaptive 0.35 logic)
# ==========================================
with tabs[0]:
    st.header("Bee Counter")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up1")
    if file:
        img = process_image_memory_safe(file)
        st.image(img, width=zoom_val)
        
        if st.button("🚀 Run Bee Detection", key="btn1"):
            img_cv = np.array(img)
            
            # Pre-scan for density
            pre_res = bee_model(img, conf=0.20, verbose=False)[0]
            count = len(pre_res.boxes)
            
            # Logic: < 10 bees (0.35 conf) | >= 10 bees (0.20 conf)
            final_conf = 0.35 if count < 10 else 0.20
            
            results = bee_model(img, conf=final_conf, verbose=False)[0]
            results.names = {i: "Bee" for i in range(len(results.names))}
            
            ann_img = results.plot(img=img_cv.copy(), line_width=1, font_size=10)
            st.subheader(f"📊 Results: {len(results.boxes)} Bees Found (Confidence: {final_conf})")
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download", get_image_download(ann_img), "bee_detection.jpg")

# ==========================================
# 2. BEE SPECIES ID (Highest Confidence + Nepal Info)
# ==========================================
with tabs[1]:
    st.header("Bee Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up2")
    if file:
        img = process_image_memory_safe(file)
        st.image(img, width=zoom_val)
        
        if st.button("🧬 Identify Primary Species", key="btn2"):
            results = bee_model(img, conf=0.20, verbose=False)[0]
            if len(results.boxes) > 0:
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                top = results[int(best_idx)]
                species_name = top.names[int(top.boxes.cls[0])]
                
                st.success(f"### Detected Species: {species_name}")
                st.image(top.plot(line_width=1, font_size=10), width=zoom_val)
                
                # Show Detailed Profile
                profile_html = BEE_PROFILES.get(species_name, "Species details coming soon.")
                st.markdown(profile_html, unsafe_allow_html=True)
                
            else: st.warning("No bees detected for identification.")

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
            
            ann_img = results.plot(img=img_cv.copy(), line_width=1, font_size=10)
            count = len(results.boxes)
            st.subheader(f"📊 Results: {count} Pests Found")
            
            if count > 3:
                st.error("🚨 ALERT: Invasion Activity!")
                trigger_vibration()
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download", get_image_download(ann_img), "pest_detection.jpg")

# ==========================================
# 4. PEST SPECIES ID (Highest Conf)
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
                st.warning(f"### Detected Threat: {top.names[int(top.boxes.cls[0])]} (Conf: {top.boxes.conf[0]:.2f})")
                st.image(top.plot(line_width=1, font_size=10), width=zoom_val)
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
                
                f_plot = res.plot(line_width=1, font_size=10)
                cv2.putText(f_plot, f"Total Unique: {len(u_ids)}", (40, 50), 1, 1.5, (255,255,255), 2)
                
                out.write(f_plot)
                vid_holder.image(cv2.cvtColor(f_plot, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release(); out.release()
            st.success(f"Summary: {len(u_ids)} unique {mode} tracked.")
            with open(t_out.name, 'rb') as f_v: st.download_button("📥 Download Video", f_v, "tracked_hive.mp4")

# --- FOOTER ---
st.markdown('<p class="footer">Developed by - Sandesh Subedi</p>', unsafe_allow_html=True)
