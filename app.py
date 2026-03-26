import os
# Fix for Streamlit Cloud settings warning and concurrency
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '500m' 

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components
import tempfile
import io
import base64
import time
import gc

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bee & Pest Health Monitor", layout="wide", page_icon="🐝")

# --- SIDEBAR DISPLAY ---
st.sidebar.header("🖼️ Display Settings")
zoom_val = st.sidebar.slider("Image Scale (Zoom)", min_value=300, max_value=2000, value=800)

st.sidebar.header("🎛️ Detection Settings")
conf_val = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=1.00, value=0.25, step=0.05)


# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs[data-baseweb="tab-list"] { gap: 10px; }
    .stTabs[data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 10px; padding: 8px 20px; font-weight: bold;
    }
    .stTabs[aria-selected="true"] { background-color: #ffc107 !important; color: black !important; }
    .stButton>button { 
        width: 100%; border-radius: 10px; font-weight: bold; height: 3.5em; 
        background-color: #ffc107; color: black; border: 2px solid #b38600; 
    }
    .stImage > img { max-height: 75vh; object-fit: contain; display: block; margin: auto; }
    .footer { text-align: center; padding: 20px; font-weight: bold; color: #5a4609; border-top: 1px solid #ddd; margin-top: 50px;}
    .bee-info { background-color: #fff9e6; padding: 20px; border-radius: 15px; border-left: 5px solid #ffc107; margin-top: 20px;}
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

# *** MEMORY FIX: Reduced max_inference_size to 512 for static images ***
def process_image_memory_safe(file, max_inference_size=512):
    """Loads, converts, and resizes image for safe processing."""
    img = Image.open(file).convert("RGB")
    
    if max(img.size) > max_inference_size:
        img.thumbnail((max_inference_size, max_inference_size), Image.Resampling.LANCZOS)
        
    return img

def get_image_download(img_array):
    img_pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return buf.getvalue()

# --- INITIALIZE ---
bee_model, enemy_model = load_models()

st.title("🐝 Hive Health & Security Monitor")

tabs = st.tabs(["🔍 Bee Detector", "🧬 Bee Species ID & Info", "🛡️ Pest Detector", "🦠 Pest Species ID", "🎥 Video Tracking"])

# ==========================================
# 1. BEE DETECTOR (Standalone)
# ==========================================
with tabs[0]:
    st.header("Bee Detector")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up1")
    
    if file:
        # Use the smallest size (512) for inference processing
        img = process_image_memory_safe(file, max_inference_size=512) 
        st.image(img, width=zoom_val)
        
        if st.button("🚀 Run Detection", key="btn1"):
            img_cv = np.array(img)
            
            # Use imgsz=512 to match the loaded image size
            results_bee = bee_model(img, conf=conf_val, imgsz=512, verbose=False)[0]
            results_bee.names = {i: "Bee" for i in range(len(results_bee.names))}
            
            ann_img = results_bee.plot(img=img_cv.copy(), line_width=1, font_size=10)
            
            st.subheader(f"📊 Results: {len(results_bee.boxes)} Bees Found")
            
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download Result", get_image_download(ann_img), "bee_detection.jpg")
            
            del img, img_cv, results_bee, ann_img
            gc.collect()
            gc.collect() 

# ==========================================
# 2. BEE SPECIES ID & INFO (Restructured)
# ==========================================
with tabs[1]:
    st.header("Bee Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up2")
    
    if 'detected_species' not in st.session_state:
        st.session_state.detected_species = None
        
    if file:
        img = process_image_memory_safe(file, max_inference_size=512)
        st.image(img, width=zoom_val)
        
        if st.button("🧬 Identify Primary Species", key="btn2"):
            results = bee_model(img, conf=conf_val, imgsz=512, verbose=False)[0]
            
            if len(results.boxes) > 0:
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                top = results[int(best_idx)]
                species_name = top.names[int(top.boxes.cls[0])]
                
                st.session_state.detected_species = species_name
                st.success(f"### Identified Species: {species_name} (Confidence: {top.boxes.conf[0]:.2f})")
                
            else: 
                st.session_state.detected_species = None
                st.warning("No bees detected for identification at this confidence level.")
            
            del img, results
            gc.collect()
            gc.collect()

    # --- INFO DISPLAY SECTION ---
    st.markdown("---")
    st.subheader("More Information of this Species")
    
    species_to_show = st.session_state.detected_species
    
    # Allow user to select species manually if no image was processed
    if not species_to_show:
        species_to_show = st.selectbox(
            "Or select a species manually:", 
            options=[""] + list(BEE_PROFILES.keys()),
            index=0
        )
        
    if species_to_show and species_to_show in BEE_PROFILES:
        profile_html = BEE_PROFILES[species_to_show]
        st.markdown(profile_html, unsafe_allow_html=True)
    elif species_to_show:
        st.info(f"Profile for {species_to_show} is available but not currently loaded.")

# ==========================================
# 3. PEST DETECTOR 
# ==========================================
with tabs[2]:
    st.header("Bee Enemy Detector")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up3")
    if file:
        img = process_image_memory_safe(file, max_inference_size=512)
        st.image(img, width=zoom_val)
        
        if st.button("🛡️ Run Security Scan", key="btn3"):
            img_cv = np.array(img)
            results = enemy_model(img, conf=0.65, imgsz=512, verbose=False)[0] 
            results.names = {i: "Pest" for i in range(len(results.names))}
            
            ann_img = results.plot(img=img_cv.copy(), line_width=1, font_size=10)
            count = len(results.boxes)
            st.subheader(f"📊 Results: {count} Pests Found")
            
            if count > 3:
                st.error("🚨 ALERT: Invasion Activity!")
                trigger_vibration()
            st.image(ann_img, width=zoom_val)
            st.download_button("📥 Download", get_image_download(ann_img), "pest_detection.jpg")
            
            del img, img_cv, results, ann_img
            gc.collect()
            gc.collect()

# ==========================================
# 4. PEST SPECIES ID 
# ==========================================
with tabs[3]:
    st.header("Pest Species Identification")
    file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'], key="up4")
    if file:
        img = process_image_memory_safe(file, max_inference_size=512)
        st.image(img, width=zoom_val)
        
        if st.button("🦠 Identify Primary Pest", key="btn4"):
            results = enemy_model(img, conf=0.65, imgsz=512, verbose=False)[0]
            if len(results.boxes) > 0:
                best_idx = np.argmax(results.boxes.conf.cpu().numpy())
                top = results[int(best_idx)]
                st.warning(f"### Detected Threat: {top.names[int(top.boxes.cls[0])]} (Conf: {top.boxes.conf[0]:.2f})")
                st.image(top.plot(line_width=1, font_size=10), width=zoom_val)
            else: 
                st.info("No threats identified.")
                
            del img, results
            gc.collect()
            gc.collect()

# ==========================================
# 5. VIDEO TRACKING (Slow but Accurate)
# ==========================================
with tabs[4]:
    st.header("Video Tracking")
    mode = st.radio("Target:",["Bees", "Pests"], horizontal=True)
    v_file = st.file_uploader("Upload Video", type=['mp4','mov','avi'])
    
    if v_file:
        if st.button("🎥 Start Tracking"):
            
            track_conf = conf_val if mode == "Bees" else 0.65
            model = bee_model if mode == "Bees" else enemy_model
            
            VIDEO_FRAME_SIZE = 512 # Keep video frames small for speed
            
            t_in_path = None
            t_out_path = None
            h264_path = None
            
            try:
                t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_in_path = t_in.name
                t_in.write(v_file.read())
                t_in.close()
                
                cap = cv2.VideoCapture(t_in_path)
                
                w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps): fps = 30 
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if max(w_orig, h_orig) > VIDEO_FRAME_SIZE:
                    scale = VIDEO_FRAME_SIZE / float(max(w_orig, h_orig))
                    w_out, h_out = int(w_orig * scale), int(h_orig * scale)
                else:
                    w_out, h_out = w_orig, h_orig

                t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_out_path = t_out.name
                t_out.close()
                
                out = cv2.VideoWriter(t_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_out, h_out))
                
                frame_count = 0
                total_sum = 0 
                
                progress_bar = st.progress(0, text="Processing video for multiple users. Please wait...")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    
                    if (w_out, h_out) != (w_orig, h_orig):
                        frame = cv2.resize(frame, (w_out, h_out))
                    
                    # Process every frame (for best accuracy) at the small size
                    res = model(frame, conf=track_conf, imgsz=VIDEO_FRAME_SIZE, verbose=False)[0] 
                    res.names = {i: mode[:-1] for i in range(len(res.names))}
                    
                    total_sum += len(res.boxes)
                    
                    f_plot = res.plot(line_width=1, font_size=10)
                    cv2.putText(f_plot, f"Total Sum of {mode}: {total_sum}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    out.write(f_plot)
                    
                    # Yield GIL occasionally
                    if frame_count % 10 == 0: 
                        time.sleep(0.001) 
                        if total_frames > 0:
                            progress_bar.progress(min(frame_count / total_frames, 1.0), text=f"Processing Video... Frame {frame_count}/{total_frames}")
                        
                cap.release()
                out.release()
                progress_bar.empty()
                
                st.success(f"🐝 Final Summary: A total sum of {total_sum} {mode} detections were recorded across the entire video.")
                
                h264_path = t_out_path.replace('.mp4', '_h264.mp4')
                os.system(f"ffmpeg -y -i {t_out_path} -vcodec libx264 -preset veryfast -crf 23 {h264_path} > /dev/null 2>&1")
                
                final_path = h264_path if os.path.exists(h264_path) else t_out_path
                
                with open(final_path, 'rb') as f_v: 
                    video_bytes = f_v.read()
                
                b64 = base64.b64encode(video_bytes).decode()
                st.markdown(f'''
                    <div style="display:flex; justify-content:center; margin-bottom: 20px;">
                        <video width="{zoom_val}" controls autoplay loop>
                            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                        </video>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 Download Tracked Video", 
                    data=video_bytes, 
                    file_name=f"tracked_{mode.lower()}.mp4", 
                    mime="video/mp4"
                )
                
            finally:
                if t_in_path and os.path.exists(t_in_path): os.remove(t_in_path)
                if t_out_path and os.path.exists(t_out_path): os.remove(t_out_path)
                if h264_path and os.path.exists(h264_path): os.remove(h264_path)
                
                gc.collect() 
                gc.collect() 

# --- FOOTER ---
st.markdown('<p class="footer">Developed by - Sandesh Subedi</p>', unsafe_allow_html=True)
