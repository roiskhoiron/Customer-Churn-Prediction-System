from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

# Load model artifacts
base_path = os.path.join(os.path.dirname(__file__), "..", "membangun_model", "models")
model_path = os.path.join(base_path, "churn_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")
features_path = os.path.join(base_path, "feature_names.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
    raise RuntimeError("Model artifacts not found. Ensure modelling.py has been run.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
ERROR_COUNT = Counter('prediction_errors_total', 'Total prediction errors')

class PredictRequest(BaseModel):
    data: dict

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
@REQUEST_LATENCY.time()
def predict(req: PredictRequest):
    REQUEST_COUNT.inc()
    try:
        # Ensure all required features are present
        input_df = pd.DataFrame([req.data])
        missing = set(feature_names) - set(input_df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing)}")
        # Align column order
        input_df = input_df[feature_names]
        # Scale only numeric columns as expected by the scaler
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChargesPerMonth", "TenureBin", "SeniorPartner"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        input_scaled = input_df
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0, 1]
        return {"prediction": int(pred), "probability": float(prob)}
    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))
