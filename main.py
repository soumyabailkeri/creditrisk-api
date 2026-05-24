from fastapi import FastAPI, Depends
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session
import pickle
import numpy as np
from database import SessionLocal, PredictionLog

app = FastAPI()

# Load model once at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class LoanApplication(BaseModel):
    applicant_name: str
    age: int
    income: float
    loan_amount: float
    employment_years: int

    @field_validator('age')
    def age_must_be_positive(cls, v):
        if v < 18:
            raise ValueError('Applicant must be at least 18 years old')
        return v

    @field_validator('loan_amount')
    def loan_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Loan amount must be greater than zero')
        return v

class RiskResponse(BaseModel):
    applicant: str
    loan_amount: float
    risk_score: float
    risk_label: str
    message: str

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/loan/apply", response_model=RiskResponse)
def apply_for_loan(application: LoanApplication, db: Session = Depends(get_db)):
    # Run ML model
    features = np.array([[
        application.age,
        application.income,
        application.loan_amount,
        application.employment_years
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    risk_score = round(float(probability[1]) * 100, 2)
    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"

    # Log to database
    log = PredictionLog(
        applicant_name=application.applicant_name,
        age=application.age,
        income=application.income,
        loan_amount=application.loan_amount,
        employment_years=application.employment_years,
        risk_score=risk_score,
        risk_label=risk_label
    )
    db.add(log)
    db.commit()

    return {
        "applicant": application.applicant_name,
        "loan_amount": application.loan_amount,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "message": f"Application assessed. Risk score: {risk_score}%"
    }

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    return [
        {
            "id": log.id,
            "applicant": log.applicant_name,
            "risk_score": log.risk_score,
            "risk_label": log.risk_label,
            "created_at": log.created_at
        }
        for log in logs
    ]