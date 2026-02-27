import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

# --- 1. MODEL CONFIGURATION ---
@st.cache_resource 
def load_my_model():
    file_id = '1_m-dKwX8uDsUSKBW8a0xZbfPAndyApcy'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'trained_model.h5'
    
    if not os.path.exists(output):
        with st.spinner("Downloading AI Model from Drive..."):
            gdown.download(url, output, quiet=False)
    
    return tf.keras.models.load_model(output)

model = load_my_model()

# --- 2. LABELS & CURES ---
# IMPORTANT: These must be in the order your model was trained!
# If your model has 38 classes, add them all here in order.
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 3. WEB INTERFACE ---
st.set_page_config(page_title="Scan Crop AI", page_icon="🌱")
st.title("🌱 Scan Crop AI: Detect & Cure")
st.markdown("Upload a leaf image or use the camera to identify diseases.")

# Tab Selection
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Live Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

with tab2:
    camera_file = st.camera_input("Take a photo")

input_file = uploaded_file if uploaded_file else camera_file

# --- 4. PREDICTION LOGIC ---
if input_file:
    img = Image.open(input_file).convert('RGB')
    st.image(img, caption="Scanning Leaf...", use_column_width=True)
    
    # Preprocessing
    img_resized = img.resize((224, 224)) # Adjust to 256, 256 if needed
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    disease_name = CLASS_NAMES[result_index]
    confidence = np.max(predictions) * 100
    
    st.success(f"### Prediction: {disease_name.replace('___', ' ').replace('_', ' ')}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Simple Cure Logic
    st.info("💡 **Recommended Action:** Use organic fungicides, ensure proper sunlight, and remove infected leaves to prevent spreading.")