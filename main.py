from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Dict
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crane Safety Predictor",
    description="API for predicting crane operation safety",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this appropriately in production
)

# Load model and scaler
try:
    model = joblib.load('crane_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

class CraneData(BaseModel):
    max_load: float = Field(..., gt=0, description="Maximum load capacity in kg")
    radius: float = Field(..., gt=0, description="Operating radius in meters")
    wind_tolerance: float = Field(..., gt=0, description="Wind tolerance in km/h")
    load_weight: float = Field(..., gt=0, description="Current load weight in kg")
    wind_speed: float = Field(..., ge=0, description="Current wind speed in km/h")

@app.post("/predict", response_model=Dict[str, str])
async def predict_safety(data: CraneData):
    try:
        # Prepare input data
        input_data = np.array([[
            data.max_load,
            data.radius,
            data.wind_tolerance,
            data.load_weight,
            data.wind_speed
        ]])
        
        # Scale input
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Log prediction
        logger.info(f"Prediction made for input: {data.dict()}")
        
        return {
            "prediction": "Safe" if prediction == 1 else "Not Safe"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 