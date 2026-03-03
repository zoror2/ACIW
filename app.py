import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Milk Purity Detector", page_icon="🥛")

# 2. Asset Loader
@st.cache_resource
def load_assets():
    try:
        # These filenames must match exactly what is in your GitHub repo
        model = joblib.load('milk_random_forest_model.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, encoder = load_assets()

# 3. Main UI
st.title("🥛 Milk Purity Checker")
st.write("Enter the numerical intensity values from your sensors to analyze the milk sample.")

if model and encoder:
    with st.form("input_form"):
        st.subheader("Sensor Input Data")
        
        # Create 2 columns for a cleaner layout
        col1, col2 = st.columns(2)
        
        with col1:
            v1 = st.number_input("Sensor 1 (Violet - 410nm)", value=0.0, format="%.2f")
            v2 = st.number_input("Sensor 2 (Blue - 435nm)", value=0.0, format="%.2f")
            v3 = st.number_input("Sensor 3 (Blue - 460nm)", value=0.0, format="%.2f")
            v4 = st.number_input("Sensor 4 (Cyan - 485nm)", value=0.0, format="%.2f")
            v5 = st.number_input("Sensor 5 (Green - 510nm)", value=0.0, format="%.2f")
            
        with col2:
            v6 = st.number_input("Sensor 6 (Green - 535nm)", value=0.0, format="%.2f")
            v7 = st.number_input("Sensor 7 (Yellow - 560nm)", value=0.0, format="%.2f")
            v8 = st.number_input("Sensor 8 (Yellow - 585nm)", value=0.0, format="%.2f")
            v9 = st.number_input("Sensor 9 (Orange - 610nm)", value=0.0, format="%.2f")
        
        submit = st.form_submit_button("Analyze Sample")

    if submit:
        # THE FIX: Use NumPy array instead of DataFrame to bypass column name validation
        # This works regardless of what names were used during training
        input_data = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9]])
        
        try:
            # Perform Prediction (raw NumPy array - no metadata)
            prediction = model.predict(input_data)
            result = encoder.inverse_transform(prediction)[0]
            
            # Display Results
            st.divider()
            if result.lower() == "pure":
                st.success(f"### Result: {result} Milk ✅")
                st.balloons()
            else:
                st.error(f"### Result: {result} Detected ⚠️")
                st.info("The model suggests an impurity based on the spectral pattern.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Note: Ensure your model was trained on exactly 9 sensors.")

else:
    st.warning("Model files not found. Please ensure 'milk_random_forest_model.pkl' and 'label_encoder.pkl' are in your GitHub repo.")