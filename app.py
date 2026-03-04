import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from scipy.spatial import distance
import streamlit.components.v1 as components
import tempfile
import io
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bee Health & Security Monitor", layout="wide", page_icon="🐝")

# --- CUSTOM CSS FOR TABS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        display: flex;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fdf6e3;
        border: 2px solid #e0c879;
        border-radius: 12px;
        padding: 10px 25px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        font-weight: 800;
        font-size: 18px;
        color: #5a4609;
        transition: all 0.3s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffc107 !important; 
        color: #000 !important;
        border: 2px solid #b38600 !important;
        transform: scale(1.05);
    }
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] { flex-direction: column; }
        .stTabs [data-baseweb="tab"] { width: 100%; text-align: center; }
    }
</style>
""", unsafe_allow_html=True)

# --- HELPERS ---

def trigger_vibration():
    """Triggers mobile vibration pattern."""
    components.html("""
        <script>
            if ("vibrate" in navigator) {
                navigator.vibrate([500, 200, 500, 200, 1000]);
            }
        </script>
    """, height=0, width=0)

@st.cache_resource
def load_models():
    """Loads YOLO models with robust path handling for Streamlit Cloud."""
    # This gets the absolute path of the current script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    bee_path = os.path.join(base_dir, 'models', 'bee_best.pt')
    enemy_path = os.path.join(base_dir, 'models', 'enemy_best.pt')
    
    # Validation check
    if not os.path.exists(bee_path) or not os.path.exists(enemy_path):
        st.error(f"Critical Error: Model files not found in {os.path.join(base_dir, 'models')}")
        st.stop()
        
    return YOLO(bee_path), YOLO(enemy_path)

def get_spatial_smoothing(results):
    """Adjusts classes based on nearest neighbor majority."""
    if not results or len(results[0].boxes) < 3:
        return results
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    coords = results[0].boxes.xywh.cpu().numpy()[:, :2] 
    
    new_classes = classes.copy()
    dist_matrix = distance.cdist(coords, coords, 'euclidean')

    for i in range(len(new_classes)):
        nn_indices = dist_matrix[i].argsort()[1:4] 
        neighbor_classes = classes[nn_indices]
        if len(set(neighbor_classes)) == 1 and neighbor_classes[0] != classes[i]:
            new_classes[i] = neighbor_classes[0]
            
    results[0].boxes.cls = new_classes
    return results

# --- INITIALIZE ---
bee_model, enemy_model = load_models()
IMG_FORMATS = ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff']
VID_FORMATS = ['mp4', 'mov', 'avi', 'mkv']

st.title("🐝 Bee Health & Security Monitor")
st.markdown("---")

tabs = st.tabs(["🔍 1. Surveillance", "🧬 2. Species Classification", "🎥 3. Video Mode"])

# ==========================================
# TAB 1: SURVEILLANCE
# ==========================================
with tabs[0]:
    st.header("Surveillance: Bee & Pest Counter")
    file_t1 = st.file_uploader("Upload Image for Hive Scan", type=IMG_FORMATS, key="uploader_t1")
    
    if file_t1:
        img = Image.open(file_t1).convert("RGB")
        img_cv = np.array(img)
        
        # Initial scan to determine density
        pre_bee = bee_model(img, conf=0.25, verbose=False)[0]
        density_count = len(pre_bee.boxes)
        
        # Logic: Adjust confidence based on bee count
        if density_count < 5:
            c_val, l_wid = 0.50, 3  # High confidence, thick lines
        else:
            c_val, l_wid = 0.15, 1  # Low confidence, thin lines
            
        # Final Processing
        res_bee = bee_model(img, conf=c_val, verbose=False)[0]
        res_pest = enemy_model(img, conf=c_val, verbose=False)[0]
        
        # Rename classes for the final report download
        res_bee.names = {k: "Bee" for k in res_bee.names}
        res_pest.names = {k: "Bee Enemy" for k in res_pest.names}
        
        # Create Annotated Image (for download only)
        ann_img = res_bee.plot(img=img_cv.copy(), line_width=l_wid)
        ann_img = res_pest.plot(img=ann_img, line_width=l_wid)
        ann_pil = Image.fromarray(ann_img)
        
        # UI: Show clean image
        st.subheader("Hive View (Original)")
        st.image(img, use_container_width=True)
        
        # Stats
        b_count = len(res_bee.boxes)
        p_count = len(res_pest.boxes)
        
        st.markdown("### 📊 Detection Report")
        col_a, col_b = st.columns(2)
        col_a.metric("Bees Found", b_count)
        col_b.metric("Pests Found", p_count)
        
        if p_count > 5:
            st.error("🔴 WARNING: High Pest Contamination! Immediate hive inspection required.")
            trigger_vibration()
        
        # Download preparation
        buf_img = io.BytesIO()
        ann_pil.save(buf_img, format="JPEG")
        
        report_text = f"BEE MONITORING REPORT\n--------------------\nTotal Bees: {b_count}\nTotal Pests: {p_count}"
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.download_button("📥 Download Annotated Image", buf_img.getvalue(), "hive_analysis.jpg", "image/jpeg")
        c2.download_button("📝 Download Text Report", report_text, "hive_report.txt", "text/plain")

# ==========================================
# TAB 2: SPECIES CLUSTERS
# ==========================================
with tabs[1]:
    st.header("Species Classification & Filtering")
    file_t2 = st.file_uploader("Upload Image for Species ID", type=IMG_FORMATS, key="uploader_t2")
    
    if file_t2:
        img = Image.open(file_t2).convert("RGB")
        img_cv = np.array(img)
        
        # Run Detection (0.20 confidence as requested)
        res_bee = bee_model(img, conf=0.20, verbose=False)[0]
        res_pest = enemy_model(img, conf=0.20, verbose=False)[0]
        
        # Apply Smoothing
        res_bee = get_spatial_smoothing([res_bee])[0]
        res_pest = get_spatial_smoothing([res_pest])[0]
        
        # Extract Species names
        bee_species = set([res_bee.names[int(c)] for c in res_bee.boxes.cls.cpu().numpy()])
        pest_species = set([res_pest.names[int(c)] for c in res_pest.boxes.cls.cpu().numpy()])
        
        if not bee_species and not pest_species:
            st.info("No organisms detected.")
        else:
            if bee_species: st.success(f"🐝 **Bee Species:** {', '.join(bee_species)}")
            if pest_species: st.warning(f"🪳 **Pest Species:** {', '.join(pest_species)}")
            
            # Show Annotated
            ann_img = res_bee.plot(img=img_cv.copy())
            ann_img = res_pest.plot(img=ann_img)
            st.image(ann_img, caption="Classification & Spatial Clustering Result", use_container_width=True)

# ==========================================
# TAB 3: VIDEO TRACKING
# ==========================================
with tabs[2]:
    st.header("Video Tracking: Joint Detection Mode")
    v_file = st.file_uploader("Upload Hive Video", type=VID_FORMATS, key="uploader_t3")
    
    if v_file:
        # Create temporary file for the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_in:
            t_in.write(v_file.read())
            in_path = t_in.name
            
        cap = cv2.VideoCapture(in_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Temp output path
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        video_ui = st.empty()
        prog_bar = st.progress(0)
        
        unique_bees = set()
        unique_pests = set()
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run Tracking (persist=True maintains IDs)
            track_bee = bee_model.track(frame, persist=True, verbose=False)
            track_pest = enemy_model.track(frame, persist=True, verbose=False)
            
            # Collect Unique IDs
            if track_bee[0].boxes.id is not None:
                unique_bees.update(track_bee[0].boxes.id.cpu().numpy().astype(int))
            if track_pest[0].boxes.id is not None:
                unique_pests.update(track_pest[0].boxes.id.cpu().numpy().astype(int))
                
            # Plot frame
            f_plot = track_bee[0].plot()
            f_plot = track_pest[0].plot(img=f_plot)
            out_writer.write(f_plot)
            
            # Every 5th frame, update the UI preview to keep it fast
            if frame_count % 5 == 0:
                video_ui.image(cv2.cvtColor(f_plot, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            frame_count += 1
            prog_bar.progress(min(frame_count / total_f, 1.0))
            
        cap.release()
        out_writer.release()
        
        st.success("✅ Tracking Finished!")
        st.markdown(f"**Total Unique Bees Tracked:** {len(unique_bees)}")
        st.markdown(f"**Total Unique Pests Tracked:** {len(unique_pests)}")
        
        with open(out_path, 'rb') as f_vid:
            st.download_button("📥 Download Processed Video", f_vid.read(), "bee_tracking.mp4", "video/mp4")
            
        # Cleanup
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)
