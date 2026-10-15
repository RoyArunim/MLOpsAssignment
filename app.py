import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Load the model globally so it's ready when the API starts
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")

# 2. Initialize FastAPI
app = FastAPI(title="Heart Disease Risk Prediction API")

# 3. Define the expected JSON input schema
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