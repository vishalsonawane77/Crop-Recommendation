import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load model and encoder ---
@st.cache_resource
def load_model():
    try:
        with open('crop_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, le
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, le = load_model()

# --- Custom CSS for better UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .header {
        color: #2e7d32;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
    }
    .input-label {
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🌾 Smart Crop Recommendation System")
st.markdown("""
    <p class="header">
    Get personalized crop recommendations based on soil and weather conditions to maximize your agricultural yield.
    </p>
""", unsafe_allow_html=True)

# --- Input Form ---
with st.form(key="crop_form"):
    st.subheader("🌱 Soil & Weather Parameters")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="input-label">🧪 Nitrogen (N) ratio</p>', unsafe_allow_html=True)
        N = st.number_input("", 0, 140, 50, key="n_input", 
                      help="Amount of Nitrogen in the soil (0-140 kg/ha)", label_visibility="collapsed")
        
        st.markdown('<p class="input-label">🧪 Phosphorus (P) ratio</p>', unsafe_allow_html=True)
        P = st.number_input("", 5, 145, 50, key="p_input",
                     help="Amount of Phosphorus in the soil (5-145 kg/ha)", label_visibility="collapsed")
        
        st.markdown('<p class="input-label">🧪 Potassium (K) ratio</p>', unsafe_allow_html=True)
        K = st.number_input("", 5, 205, 50, key="k_input",
                     help="Amount of Potassium in the soil (5-205 kg/ha)", label_visibility="collapsed")
        
        st.markdown('<p class="input-label">⚗️ Soil pH Level</p>', unsafe_allow_html=True)
        ph = st.number_input("", 3.0, 10.0, 6.5, 0.1, key="ph_input",
                      help="Soil pH level (3.0-10.0)", label_visibility="collapsed")

    with col2:
        st.markdown('<p class="input-label">🌡️ Temperature (°C)</p>', unsafe_allow_html=True)
        temperature = st.number_input("", 0.0, 50.0, 25.0, 0.1, key="temp_input",
                               help="Average temperature (0-50°C)", label_visibility="collapsed")
        
        st.markdown('<p class="input-label">💧 Humidity (%)</p>', unsafe_allow_html=True)
        humidity = st.number_input("", 10.0, 100.0, 70.0, 0.1, key="humidity_input",
                           help="Relative humidity percentage (10-100%)", label_visibility="collapsed")
        
        st.markdown('<p class="input-label">🌧️ Rainfall (mm)</p>', unsafe_allow_html=True)
        rainfall = st.number_input("", 0.0, 300.0, 100.0, 1.0, key="rainfall_input",
                            help="Expected rainfall (0-300 mm)", label_visibility="collapsed")
    
    # Form submit button
    submitted = st.form_submit_button("🔍 Get Crop Recommendation")

# --- Prediction and Results ---
if submitted:
    # Validate inputs
    if not all([N, P, K, temperature, humidity, ph, rainfall]):
        st.warning("⚠️ Please fill in all fields")
        st.stop()
    
    with st.spinner("🔮 Analyzing conditions..."):
        try:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)[0]
            crop_name = le.inverse_transform([prediction])[0]
            
            # Display results
            st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Recommended Crop: <strong>{crop_name.upper()}</strong></h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Show crop image if available
            try:
                image_path = f"crop_images/{crop_name.lower()}.jpg"
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption=crop_name, use_container_width=True)
                else:
                    st.info(f"ℹ️ No preview available for {crop_name}")
            except Exception as e:
                st.warning(f"Couldn't load crop image: {e}")
            
            # Additional information section
            with st.expander("ℹ️ About this recommendation"):
                st.write(f"""
                This recommendation is based on the following conditions:
                - Nitrogen: {N} kg/ha
                - Phosphorus: {P} kg/ha
                - Potassium: {K} kg/ha
                - Temperature: {temperature}°C
                - Humidity: {humidity}%
                - pH Level: {ph}
                - Rainfall: {rainfall} mm
                
                The model predicts that **{crop_name}** would be the most suitable crop 
                for these conditions.
                """)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# --- Reset button ---
if submitted:
    if st.button("🔄 Make Another Prediction"):
        st.experimental_rerun()

# --- Footer ---
st.markdown("""
    <div class="footer">
        <hr>
        <p>Agricultural Decision Support System • Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)