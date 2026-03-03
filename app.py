import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Milk Adulteration Detector", page_icon="🥛")

# 2. Model Loader (Optimized)
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

# 3. Sidebar Information
st.sidebar.title("About the Project")
st.sidebar.info(
    "This AI model uses Random Forest to detect milk adulteration "
    "based on optical sensor readings (400nm - 1100nm)."
)

# 4. Main UI
st.title("🥛 Milk Purity Checker")
st.write("Enter the intensity values from your optical sensors below to analyze the sample.")

if model and encoder:
    with st.form("input_form"):
        st.subheader("Sensor Input (Reflectance Intensity)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            v1 = st.number_input("410nm (Violet)", value=0.0, format="%.2f")
            v2 = st.number_input("435nm (Blue)", value=0.0, format="%.2f")
            v3 = st.number_input("460nm (Blue)", value=0.0, format="%.2f")
            v4 = st.number_input("485nm (Cyan)", value=0.0, format="%.2f")
            
        with col2:
            v5 = st.number_input("510nm (Green)", value=0.0, format="%.2f")
            v6 = st.number_input("535nm (Green)", value=0.0, format="%.2f")
            v7 = st.number_input("560nm (Yellow)", value=0.0, format="%.2f")
            v8 = st.number_input("585nm (Yellow)", value=0.0, format="%.2f")
        
        submit = st.form_submit_button("Analyze Sample")

    if submit:
        # --- THE FIX: EXACT COLUMN NAMES ---
        # If your Kaggle print(X_train.columns) showed different names, 
        # replace these strings with THOSE exact strings.
        feature_names = ['410nm', '435nm', '460nm', '485nm', '510nm', '535nm', '560nm', '585nm']
        
        # Create DataFrame
        input_df = pd.DataFrame([[v1, v2, v3, v4, v5, v6, v7, v8]], columns=feature_names)
        
        try:
            # Prediction
            prediction = model.predict(input_df)
            result = encoder.inverse_transform(prediction)[0]
            
            # Display Results
            st.divider()
            if result.lower() == "pure":
                st.success(f"### Result: {result} Milk ✅")
                st.balloons()
            else:
                st.error(f"### Result: {result} Detected ⚠️")
                st.write("The spectral pattern indicates an impurity in the sample.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Ensure the number of inputs matches what the model was trained on.")

else:
    st.warning("Please ensure 'milk_random_forest_model.pkl' and 'label_encoder.pkl' are in your GitHub repository.")