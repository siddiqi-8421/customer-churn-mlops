"""
FastAPI service that loads the trained model and serves live predictions.
Trained model ko load karke live prediction dene wali API.

Run from the project root:
    uvicorn app.main:app --port 8000
Then open http://127.0.0.1:8000/docs to test it in the browser.
"""
import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# let this file find src/features.py | is file ko src/features.py dhundhne do
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import ChurnFeatureEngineer  # needed so joblib can load the pipeline | pipeline load karne ke liye zaroori
from app.schema import CustomerFeatures, PredictionResponse  # import as part of the app package | app package ke hisse ke roop me import

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
model = None  # will hold the loaded model | load hua model yahan rakhenge


@app.on_event("startup")
def load_model():
    # runs once when the server starts | server start hote hi ek baar chalta hai
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
    model = joblib.load(MODEL_PATH)  # load the saved pipeline | saved pipeline load karo


@app.get("/health")
def health():
    # is the server alive and model loaded? | server chal raha hai aur model load hua?
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # convert request into a 1-row dataframe | request ko 1-row table banao
    row = pd.DataFrame([customer.model_dump()])

    # pipeline's first step does feature engineering automatically | pipeline ka pehla step khud feature engineering karta hai
    proba = float(model.predict_proba(row)[0, 1])  # churn probability | churn hone ka chance
    pred = int(proba >= 0.5)                         # final yes/no | final yes/no
    risk = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"  # readable label | padhne layak label

    return PredictionResponse(churn_probability=round(proba, 4), churn_prediction=pred, risk_level=risk)