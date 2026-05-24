from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://creditrisk_user:creditrisk123@localhost/creditrisk_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    applicant_name = Column(String, nullable=False)
    age = Column(Integer)
    income = Column(Float)
    loan_amount = Column(Float)
    employment_years = Column(Integer)
    risk_score = Column(Float)
    risk_label = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create table
Base.metadata.create_all(bind=engine)