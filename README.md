# Credit Risk Scoring API

A production-style machine learning API that assesses loan application risk in real time. Built with FastAPI, Scikit-learn, PostgreSQL, and Docker — designed around 3 years of hands-on BFSI domain experience.

---

## What it does

Send a loan application as JSON → get back a risk score, risk label, and confidence percentage instantly.

**Example request:**
```json
{
    "applicant_name": "Soumya",
    "age": 25,
    "income": 30000,
    "loan_amount": 10000,
    "employment_years": 2
}
```

**Example response:**
```json
{
    "applicant": "Soumya",
    "loan_amount": 10000.0,
    "risk_score": 82.5,
    "risk_label": "HIGH RISK",
    "message": "Application assessed. Risk score: 82.5%"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ML Model | Scikit-learn (RandomForestClassifier) |
| Validation | Pydantic |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Containerisation | Docker + docker-compose |
| Documentation | Auto-generated Swagger UI |

---

## Architecture
POST /loan/apply
→ Pydantic validates input (age, income, loan_amount, employment_years)
→ RandomForest model predicts default probability
→ Risk score and label generated
→ Prediction logged to PostgreSQL
→ Response returned
GET /predictions
→ Retrieves full prediction history from PostgreSQL
→ Returns all assessments with timestamps

---

## Running Locally

### Option 1 — Docker (recommended)
```bash
git clone https://github.com/soumyabailkeri/creditrisk-api.git
cd creditrisk-api
docker compose up --build
```

Visit `http://localhost:8001/docs` for interactive Swagger UI.

### Option 2 — Without Docker
```bash
git clone https://github.com/soumyabailkeri/creditrisk-api.git
cd creditrisk-api
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --port 8001
```

Requires PostgreSQL running locally. Update `DATABASE_URL` in `database.py`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Service status |
| POST | `/loan/apply` | Submit loan application for risk assessment |
| GET | `/predictions` | Retrieve all prediction history |
| GET | `/docs` | Interactive Swagger documentation |

---

## ML Model Details

- **Algorithm:** RandomForestClassifier (100 estimators)
- **Features:** Age, Annual Income, Loan Amount, Employment Years
- **Output:** Binary prediction (default/no default) + probability score
- **Training:** `train_model.py` — retrained automatically during Docker build
- **Serialisation:** pickle — model loaded once at API startup for performance

---

## Input Validation

Pydantic enforces strict validation on all inputs:
- `age` — must be 18 or older
- `loan_amount` — must be greater than zero
- Invalid types rejected with `422 Unprocessable Entity` and detailed error messages

---

## Domain Context

This project was built drawing on 3 years of professional experience in the BFSI domain at Tata Consultancy Services, where I worked on core banking APIs for loan processing, teller operations, and cash management. The risk model reflects real-world loan assessment factors used by financial institutions.

---

## Project Structure
creditrisk-api/
├── main.py              # FastAPI app and endpoints
├── database.py          # SQLAlchemy models and DB connection
├── train_model.py       # ML model training script
├── model.pkl            # Serialised trained model
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container build instructions
├── docker-compose.yml   # Multi-container orchestration
└── .gitignore           # Excludes .env and sensitive files

---

## Author

**Soumya Bailkeri**
Backend Software Engineer | Python · FastAPI · Django · ML
[LinkedIn](https://linkedin.com/in/soumyabailkeri) · [GitHub](https://github.com/soumyabailkeri)