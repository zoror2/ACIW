import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Milk Purity Checker", layout="centered")

# 1. Load Model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('milk_random_forest_model.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, encoder = load_model()

if model is None:
    st.stop()

# 2. App Title & Description
st.title("🥛 Milk Adulteration Detector")
st.write("Enter the optical sensor readings to detect impurities like Water, Starch, or Urea.")

# 3. Input Form
with st.form("prediction_form"):
    st.subheader("Sensor Readings (Intensity)")
    
    # Create 2 columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Note: These labels are generic. If your CSV had specific wavelengths (e.g. '450nm'), rename them here!
        v1 = st.number_input("Sensor 1 (e.g., Violet/400nm)", value=0.0)
        v2 = st.number_input("Sensor 2 (e.g., Blue/500nm)", value=0.0)
        v3 = st.number_input("Sensor 3 (e.g., Green/600nm)", value=0.0)
        v4 = st.number_input("Sensor 4 (e.g., Red/700nm)", value=0.0)
        
    with col2:
        v5 = st.number_input("Sensor 5 (e.g., NIR 1/800nm)", value=0.0)
        v6 = st.number_input("Sensor 6 (e.g., NIR 2/900nm)", value=0.0)
        v7 = st.number_input("Sensor 7 (e.g., NIR 3/1000nm)", value=0.0)
        # Note: Check if your model was trained on 7 or 8 columns. 
        # If 7, delete this last input. If 8, keep it.
        v8 = st.number_input("Sensor 8 (e.g., NIR 4/1100nm)", value=0.0)
    
    submitted = st.form_submit_button("Analyze Sample")

# 4. Prediction Logic
if submitted:
    # IMPORTANT: The number of inputs here must match your training data exactly.
    input_data = pd.DataFrame([[v1, v2, v3, v4, v5, v6, v7, v8]], 
                              columns=['400nm', '500nm', '600nm', '700nm', '800nm', '900nm', '1000nm', '1100nm'])
    
    # Make Prediction
    prediction_index = model.predict(input_data)
    result = encoder.inverse_transform(prediction_index)[0]

    st.divider()
    if result == "Pure":
        st.success(f"### Result: {result} Milk ✅")
        st.balloons()
    else:
        st.error(f"### Result: Contaminated with {result} ⚠️")