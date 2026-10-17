import logging
import time
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator



# Setup Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#  Load the model globally so it's ready when the API starts
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")

# Initialize FastAPI
app = FastAPI(title="Heart Disease Risk Prediction API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000 # Convert to milliseconds
    logger.info(f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Time: {process_time:.2f}ms")
    return response


Instrumentator().instrument(app).expose(app)


# Define the expected JSON input schema
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# 4. Create the prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found on server.")
    
    # Convert validated JSON payload to a Pandas DataFrame
    # Using model_dump() (Pydantic v2) to get dictionary
    df = pd.DataFrame([data.model_dump()])
    
    try:
        # Predict class (0 or 1) and get probability for class 1
        prediction = int(model.predict(df)[0])
        confidence = float(model.predict_proba(df)[0][1])
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk_status": "High Risk" if prediction == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Optional health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}