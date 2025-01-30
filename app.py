import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
from typing import Dict, List
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Crane Safety Predictor",
    page_icon="🏗️",
    layout="wide"
)

# Title and description
st.title("🏗️ Crane Safety Prediction System")
st.markdown("""
This application helps predict crane operation safety based on various parameters.
Please input the required values below or upload a CSV file for batch predictions.
""")

# API endpoint
API_ENDPOINT = "http://localhost:8000/predict"

def predict_single(data: Dict) -> str:
    """Make a single prediction using the FastAPI backend"""
    try:
        response = requests.post(API_ENDPOINT, json=data)
        response.raise_for_status()
        return response.json()["prediction"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def predict_batch(df: pd.DataFrame) -> List[str]:
    """Make predictions for multiple entries"""
    predictions = []
    for _, row in df.iterrows():
        data = {
            "max_load": float(row["max_load"]),
            "radius": float(row["radius"]),
            "wind_tolerance": float(row["wind_tolerance"]),
            "load_weight": float(row["load_weight"]),
            "wind_speed": float(row["wind_speed"])
        }
        prediction = predict_single(data)
        predictions.append(prediction)
    return predictions

# Create tabs for single prediction and batch prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single Prediction")
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        max_load = st.number_input("Maximum Load Capacity (kg)", 
                                 min_value=0.0, 
                                 value=5000.0,
                                 step=100.0)
        radius = st.number_input("Operating Radius (m)", 
                               min_value=0.0, 
                               value=15.0,
                               step=0.5)
        wind_tolerance = st.number_input("Wind Tolerance (km/h)", 
                                       min_value=0.0, 
                                       value=50.0,
                                       step=1.0)
    
    with col2:
        load_weight = st.number_input("Current Load Weight (kg)", 
                                    min_value=0.0, 
                                    value=3000.0,
                                    step=100.0)
        wind_speed = st.number_input("Current Wind Speed (km/h)", 
                                   min_value=0.0, 
                                   value=25.0,
                                   step=1.0)
    
    if st.button("Predict Safety"):
        data = {
            "max_load": max_load,
            "radius": radius,
            "wind_tolerance": wind_tolerance,
            "load_weight": load_weight,
            "wind_speed": wind_speed
        }
        
        prediction = predict_single(data)
        
        if prediction:
            color = "green" if prediction == "Safe" else "red"
            st.markdown(f"### Prediction: <span style='color:{color}'>{prediction}</span>", 
                       unsafe_allow_html=True)

with tab2:
    st.header("Batch Prediction")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ["max_load", "radius", "wind_tolerance", 
                              "load_weight", "wind_speed"]
            
            if all(col in df.columns for col in required_columns):
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Run Batch Prediction"):
                    predictions = predict_batch(df)
                    
                    # Add predictions to dataframe
                    df["prediction"] = predictions
                    
                    # Display results
                    st.write("### Prediction Results")
                    st.dataframe(df)
                    
                    # Create visualizations
                    st.write("### Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction distribution
                        fig1 = px.pie(df, 
                                    names="prediction", 
                                    title="Safety Prediction Distribution")
                        st.plotly_chart(fig1)
                    
                    with col2:
                        # Load weight vs Wind speed scatter plot
                        fig2 = px.scatter(df, 
                                        x="load_weight", 
                                        y="wind_speed",
                                        color="prediction",
                                        title="Load Weight vs Wind Speed")
                        st.plotly_chart(fig2)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="crane_safety_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error("CSV file must contain the following columns: " + 
                        ", ".join(required_columns))
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ by Your Name</p>
</div>
""", unsafe_allow_html=True) 