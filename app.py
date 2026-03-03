import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from scipy.spatial import distance

# Page Configuration
st.set_page_config(page_title="Intl Bee Research Dashboard", layout="wide")

# 1. Load your "Best" Models
@st.cache_resource
def load_models():
    # Make sure these names match your files in the models folder!
    bee_mod = YOLO('models/bee_best.pt')
    env_mod = YOLO('models/enemy_best.pt')
    return bee_mod, env_mod

bee_model, enemy_model = load_models()

# Helper Logic: Spatial Smoothing (Tab 2)
def get_spatial_smoothing(results):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    if len(boxes) < 3: return results # Need at least 3 bees to cluster
    
    new_classes = classes.copy()
    coords = results[0].boxes.xywh.cpu().numpy()[:, :2] # Get centers
    dist_matrix = distance.cdist(coords, coords, 'euclidean')

    for i in range(len(new_classes)):
        # Find 3 nearest neighbors
        nn_indices = dist_matrix[i].argsort()[1:4] 
        neighbor_classes = classes[nn_indices]
        # If neighbors are same but middle is different, overrule
        if len(set(neighbor_classes)) == 1 and neighbor_classes[0] != classes[i]:
            new_classes[i] = neighbor_classes[0]
            
    results[0].boxes.cls = new_classes
    return results

# UI Header
st.title("🐝 International Bee Health & Security Monitor")
st.markdown("---")

tabs = st.tabs(["1. Surveillance", "2. Species Clusters", "3. Pest Security", "4. Hive Health", "5. Joint Analytics"])

# --- TAB 1: SURVEILLANCE ---
with tabs[0]:
    st.header("Bee Detection")
    file = st.file_uploader("Upload Image", type=['jpg','png'], key="t1")
    if file:
        img = Image.open(file)
        results = bee_model(img)
        count = len(results[0].boxes)
        
        if count == 0:
            st.warning("No Bee Detected")
            st.image(img)
        elif count < 5:
            st.image(results[0].plot(), caption=f"Bee Count: {count}")
        else:
            st.success(f"Bee Detected! Total number: {count}")
            st.image(img, caption="High density detected.")

# --- TAB 2: SPECIES CLUSTERS ---
with tabs[1]:
    st.header("Species Classification")
    file = st.file_uploader("Upload Image", type=['jpg','png'], key="t2")
    if file:
        img = Image.open(file)
        res = bee_model(img)
        smoothed = get_spatial_smoothing(res) # Apply your cluster logic
        st.image(smoothed[0].plot(), caption="Classification with Spatial Smoothing applied")

# --- TAB 3: PEST SECURITY ---
with tabs[2]:
    st.header("Identify Pest")
    file = st.file_uploader("Upload Image", type=['jpg','png'], key="t3")
    if file:
        img = Image.open(file)
        results = enemy_model(img)
        count = len(results[0].boxes)
        
        if count > 0:
            st.error(f"Bee Enemy Found: {count}")
            if count > 5:
                st.write("🔴 **WARNING: Please take care of the hive!**")
            st.image(results[0].plot())
        else:
            st.info("No enemies found.")

# --- TAB 4: HIVE HEALTH ---
with tabs[3]:
    st.header("Hive Condition Report")
    file = st.file_uploader("Upload Image", type=['jpg','png'], key="t4")
    if file:
        img = Image.open(file)
        results = enemy_model(img)
        if len(results[0].boxes) == 0:
            st.success("✨ No enemy found. Hive is in perfect condition!")
        else:
            st.image(results[0].plot(), caption="Pest detected in hive.")

# --- TAB 5: JOINT ANALYTICS ---
with tabs[4]:
    st.header("Joint Detection & Video Mode")
    v_file = st.file_uploader("Upload Video", type=['mp4','mov','avi'])
    if v_file:
        tfile = st.empty()
        with open("temp.mp4", "wb") as f: f.write(v_file.read())
        cap = cv2.VideoCapture("temp.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Run both
            r1 = bee_model(frame)
            r2 = enemy_model(frame)
            # Combine plots
            frame = r1[0].plot()
            frame = r2[0].plot()
            tfile.image(frame, channels="BGR")
        cap.release()