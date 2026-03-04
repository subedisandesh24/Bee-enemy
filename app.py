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

# Page Configuration
st.set_page_config(page_title="Bee Health & Security Monitor", layout="wide", page_icon="🐝")

# --- CUSTOM CSS FOR EYE-CATCHY & MOBILE-FRIENDLY TABS ---
st.markdown("""
<style>
    /* Tab styling */
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
    /* Active Tab */
    .stTabs [aria-selected="true"] {
        background-color: #ffc107 !important; 
        color: #000 !important;
        border: 2px solid #b38600 !important;
        transform: scale(1.05);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ffe082;
    }
    
    /* Mobile Phone Vertical Layout */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-direction: column;
        }
        .stTabs[data-baseweb="tab"] {
            width: 100%;
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper: Vibration Trigger (Works on Android/Chrome)
def trigger_vibration():
    components.html("""
        <script>
            if ("vibrate" in navigator) {
                // Vibrate a strong SOS pattern
                navigator.vibrate([500, 200, 500, 200, 1000]);
            }
        </script>
    """, height=0, width=0)

# 1. Load Models
@st.cache_resource
def load_models():
    bee_mod = YOLO('models/bee_best.pt')
    env_mod = YOLO('models/enemy_best.pt')
    return bee_mod, env_mod

bee_model, enemy_model = load_models()

# Supported Formats
IMG_FORMATS = ['jpg', 'png', 'jpeg', 'webp', 'bmp', 'tiff', 'heic']
VID_FORMATS =['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv']

# Helper Logic: Spatial Smoothing
def get_spatial_smoothing(results):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    if len(boxes) < 3: 
        return results # Need at least 3 bees to cluster
    
    new_classes = classes.copy()
    coords = results[0].boxes.xywh.cpu().numpy()[:, :2] # Get centers
    dist_matrix = distance.cdist(coords, coords, 'euclidean')

    for i in range(len(new_classes)):
        nn_indices = dist_matrix[i].argsort()[1:4] # 3 nearest neighbors
        neighbor_classes = classes[nn_indices]
        # Overrule if middle is different from surrounding majority
        if len(set(neighbor_classes)) == 1 and neighbor_classes[0] != classes[i]:
            new_classes[i] = neighbor_classes[0]
            
    results[0].boxes.cls = new_classes
    return results

# UI Header
st.title("🐝 Bee Health & Security Monitor")
st.markdown("---")

# Setup 3 Main Tabs based on new requirements
tabs = st.tabs(["🔍 1. Surveillance", "🧬 2. Species Classification", "🎥 3. Video Mode"])

# ==========================================
# --- TAB 1: SURVEILLANCE ---
# ==========================================
with tabs[0]:
    st.header("Surveillance: Bee & Pest Counter")
    file = st.file_uploader("Upload Image", type=IMG_FORMATS, key="t1")
    
    if file:
        img = Image.open(file).convert("RGB")
        img_cv = np.array(img)
        
        # Initial scan to count
        initial_bee = bee_model(img, conf=0.25, verbose=False)[0]
        initial_pest = enemy_model(img, conf=0.25, verbose=False)[0]
        
        total_bees = len(initial_bee.boxes)
        total_pests = len(initial_pest.boxes)
        
        # Adjust confidence and bounding box size based on bee count
        if total_bees < 5:
            conf_level = 0.50  # High confidence
            line_w = 3         # Large box
        else:
            conf_level = 0.15  # Low confidence interval
            line_w = 1         # Small box
            
        # Final Run
        final_bee = bee_model(img, conf=conf_level, verbose=False)[0]
        final_pest = enemy_model(img, conf=conf_level, verbose=False)[0]
        
        # Override class names to only show "Bee" or "Bee Enemy"
        final_bee.names = {k: "Bee" for k in final_bee.names}
        final_pest.names = {k: "Bee Enemy" for k in final_pest.names}
        
        # Combine plotting onto one image array
        annotated_img = final_bee.plot(img=img_cv.copy(), line_width=line_w)
        annotated_img = final_pest.plot(img=annotated_img, line_width=line_w)
        
        # Convert back to PIL for download
        annotated_pil = Image.fromarray(annotated_img)
        
        # --- UI DISPLAY ---
        st.subheader("Original Image (Clean)")
        st.image(img, use_container_width=True) # Do not show annotated on UI as requested
        
        st.markdown(f"### 📊 Detection Report")
        st.markdown(f"- **Total Bees Detected:** {len(final_bee.boxes)}")
        st.markdown(f"- **Total Pests Detected:** {len(final_pest.boxes)}")
        
        if len(final_pest.boxes) > 5:
            st.error("🔴 WARNING: High Pest Contamination! Please safeguard the hive immediately!")
            trigger_vibration() # Vibrate device
        
        # Download logic
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        report_text = f"Total Bees: {len(final_bee.boxes)}\nTotal Pests: {len(final_pest.boxes)}"
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download Annotated Image", data=byte_im, file_name="surveillance_annotated.jpg", mime="image/jpeg")
        with col2:
            st.download_button("📝 Download Text Report", data=report_text, file_name="surveillance_report.txt", mime="text/plain")

# ==========================================
# --- TAB 2: SPECIES CLUSTERS ---
# ==========================================
with tabs[1]:
    st.header("Species Classification & Filtering")
    file = st.file_uploader("Upload Image", type=IMG_FORMATS, key="t2")
    
    if file:
        img = Image.open(file).convert("RGB")
        
        # Run with low confidence interval as requested
        res_bee = bee_model(img, conf=0.20)[0]
        res_pest = enemy_model(img, conf=0.20)[0]
        
        bee_count = len(res_bee.boxes)
        pest_count = len(res_pest.boxes)
        
        if bee_count == 0 and pest_count == 0:
            st.info("🟢 No Bee or Pest detected in the image.")
            st.image(img)
        else:
            status =[]
            if bee_count > 0: status.append("Bee")
            if pest_count > 0: status.append("Pest")
            
            st.success(f"🔍 Presence Detected: **{' and '.join(status)}**")
            
            # Apply Spatial Smoothing (cluster adjustment)
            res_bee_smoothed = get_spatial_smoothing([res_bee])[0]
            res_pest_smoothed = get_spatial_smoothing([res_pest])[0]
            
            # Extract unique species names detected
            detected_bee_species = set([res_bee_smoothed.names[int(c)] for c in res_bee_smoothed.boxes.cls.cpu().numpy()])
            detected_pest_species = set([res_pest_smoothed.names[int(c)] for c in res_pest_smoothed.boxes.cls.cpu().numpy()])
            
            if detected_bee_species:
                st.write(f"**Bee Species Found:** {', '.join(detected_bee_species)}")
            if detected_pest_species:
                st.write(f"**Pest Species Found:** {', '.join(detected_pest_species)}")
                
            # Plot
            img_cv = np.array(img)
            ann_img = res_bee_smoothed.plot(img=img_cv.copy(), line_width=2)
            ann_img = res_pest_smoothed.plot(img=ann_img, line_width=2)
            
            st.image(ann_img, caption="Classification with Spatial Clustering Logic Applied", use_container_width=True)

# ==========================================
# --- TAB 3: VIDEO TRACKING ---
# ==========================================
with tabs[2]:
    st.header("Video Tracking: Joint Detection Mode")
    v_file = st.file_uploader("Upload Video", type=VID_FORMATS, key="t3")
    
    if v_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(v_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Temp file for processed video
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_temp.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
        
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Tracking unique IDs
        unique_bees = set()
        unique_pests = set()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                
            # Run tracking
            r1 = bee_model.track(frame, persist=True, verbose=False)
            r2 = enemy_model.track(frame, persist=True, verbose=False)
            
            # Store Unique IDs
            if r1[0].boxes.id is not None:
                unique_bees.update(r1[0].boxes.id.cpu().numpy())
            if r2[0].boxes.id is not None:
                unique_pests.update(r2[0].boxes.id.cpu().numpy())
                
            # Plot frame
            plot_frame = r1[0].plot()
            plot_frame = r2[0].plot(img=plot_frame)
            
            out.write(plot_frame)
            
            # Update UI occasionally to prevent browser lag
            if frame_idx % 3 == 0:
                # Convert BGR to RGB for Streamlit rendering
                ui_frame = cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(ui_frame, channels="RGB", use_container_width=True)
            
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            
        cap.release()
        out.release()
        
        st.success("✅ Video Processing Complete!")
        st.markdown(f"### 📊 Final Tracking Report")
        st.markdown(f"- **Total Unique Bees Tracked:** {len(unique_bees)}")
        st.markdown(f"- **Total Unique Pests Tracked:** {len(unique_pests)}")
        
        # Read the saved output video for downloading
        with open(output_temp.name, 'rb') as f:
            video_bytes = f.read()
            
        st.download_button(
            label="📥 Download Processed Video",
            data=video_bytes,
            file_name="processed_tracking_video.mp4",
            mime="video/mp4"
        )
        
        # Cleanup temp files
        os.unlink(tfile.name)
        os.unlink(output_temp.name)
