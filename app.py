import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_diagnostic_model():
    # Loads the model file
    return tf.keras.models.load_model('brain_hemorrhage_model.h5')

try:
    model = load_diagnostic_model()
    class_names = ['bleed', 'normal'] # Alphabetical order
except Exception as e:
    # IF LOADING FAILS: Show error and STOP the script
    st.error(f"❌ Critical Error: Could not load model.")
    st.error(f"Details: {e}")
    st.warning("⚠️ Please check: Is 'brain_hemorrhage_model.h5' in the same folder as 'app.py'?")
    st.stop() # <--- THIS LINE PREVENTS THE CRASH
# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI", layout="wide", page_icon="🧠")

st.title("🧠 NeuroScan AI: Priority Diagnostic Hub")
st.markdown("""
    **Medical Triage System:** Upload one or multiple CT scans. 
    The system will automatically prioritize cases requiring urgent review.
""")

# --- 3. FILE UPLOADER ---
uploaded_files = st.file_uploader(
    "Upload CT Scans (Batch processing supported)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    results_list = []

    with st.spinner('Performing deep scan analysis...'):
        for file in uploaded_files:
            # Image Pre-processing
            img = Image.open(file).convert('RGB')
            resized_img = img.resize((128, 128))
            img_array = tf.keras.utils.img_to_array(resized_img)
            img_array = tf.expand_dims(img_array, 0) # Create batch axis

            # Prediction Logic
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            class_idx = np.argmax(score)
            conf = 100 * np.max(score)
            
            label = class_names[class_idx]
            
            # Store data for sorting (Priority 1 for bleed, 0 for normal)
            results_list.append({
                "filename": file.name,
                "image": img,
                "label": label,
                "confidence": conf,
                "priority": 1 if label == 'bleed' else 0
            })

    # --- 4. PRIORITY SORTING ---
    # Sort: First by Priority (Bleeds up), then by Confidence (highest first)
    sorted_results = sorted(results_list, key=lambda x: (x['priority'], x['confidence']), reverse=True)

    # --- 5. DASHBOARD DISPLAY ---
    st.divider()
    st.header("📋 Diagnostic Queue")

    for item in sorted_results:
        # Create a visual card for each scan
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.image(item['image'], use_container_width=True)
            
            with col2:
                if item['label'] == 'bleed':
                    st.error(f"🚨 **URGENT:** Hemorrhage Detected in `{item['filename']}`")
                    st.markdown("⚠️ **Action Required:** Immediate neurosurgical consultation recommended.")
                else:
                    st.success(f"✅ **NORMAL:** `{item['filename']}`")
                    st.markdown("Status: No immediate abnormalities detected by AI.")
            
            with col3:
                st.metric("Confidence", f"{item['confidence']:.1f}%")
                if item['label'] == 'bleed':
                    st.button("Mark as Reviewed", key=item['filename'])

else:
    st.info("👋 Welcome! Please upload CT scan images to begin the automated triage process.")

# --- 6. SIDEBAR INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491314.png", width=100)
    st.title("System Status")
    st.success("Model: 5-Block CNN Active")
    st.info("Input Size: 128x128 RGB")
    st.write("---")
    st.caption("Disclaimer: This tool is for educational purposes and should not be used for actual medical diagnosis.")