from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import pickle
import numpy as np

app = FastAPI()

# Load model once when server starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.post("/loan/apply", response_model=RiskResponse)
def apply_for_loan(application: LoanApplication):
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
    
    return {
        "applicant": application.applicant_name,
        "loan_amount": application.loan_amount,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "message": f"Application assessed. Risk score: {risk_score}%"
    }