"""
Defines what a valid request/response looks like for the API.
API ke liye valid request/response ka structure define karta hai.
"""
from pydantic import BaseModel  # validates incoming data automatically | aane wala data automatically check karta hai


class CustomerFeatures(BaseModel):
    # every field here must be sent by whoever calls the API | ye sab fields API call karne wale ko bhejne honge
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        # example shown in the API docs page | API docs page me example dikhega
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 250.0,
            }
        }


class PredictionResponse(BaseModel):
    # what the API sends back | API wapas kya bhejega
    churn_probability: float   # risk score between 0 and 1 | 0 se 1 ke beech risk score
    churn_prediction: int      # final 0 or 1 | final 0 ya 1
    risk_level: str            # human-readable label | insaan ke padhne layak label