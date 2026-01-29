"""
FastAPI Prediction Service for Insurance Enrollment

This module provides a REST API for serving insurance enrollment predictions
using the trained machine learning model.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /           - Health check
    GET  /info       - Model information
    POST /predict    - Single prediction
    POST /batch      - Batch predictions

Author: Data Science Team
Date: 2026-01-29
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting employee insurance enrollment likelihood",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model and processor
model = None
processor = None
model_info = {}


class EmployeeData(BaseModel):
    """Schema for employee data input."""
    age: int = Field(..., ge=18, le=100, description="Employee age (18-100)")
    gender: str = Field(..., description="Gender (Male, Female, Other)")
    marital_status: str = Field(..., description="Marital status (Single, Married, Divorced, Widowed)")
    salary: float = Field(..., gt=0, description="Annual salary")
    employment_type: str = Field(..., description="Employment type (Full-time, Part-time, Contract)")
    region: str = Field(..., description="Region (Northeast, South, Midwest, West)")
    has_dependents: str = Field(..., description="Has dependents (Yes, No)")
    tenure_years: float = Field(..., ge=0, description="Years of tenure")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "Male",
                "marital_status": "Married",
                "salary": 75000.0,
                "employment_type": "Full-time",
                "region": "West",
                "has_dependents": "Yes",
                "tenure_years": 5.5
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    enrolled: int = Field(..., description="Predicted enrollment (0 or 1)")
    probability: float = Field(..., description="Probability of enrollment")
    confidence: str = Field(..., description="Confidence level (low, medium, high)")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    employees: List[EmployeeData]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResponse]
    count: int


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    model_name: str
    feature_columns: List[str]
    model_type: str
    loaded: bool


def load_models():
    """Load the trained model and data processor from disk."""
    global model, processor, model_info
    
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_model.joblib")
    processor_path = os.path.join(models_dir, "data_processor.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(processor_path):
        raise FileNotFoundError(
            "Model files not found. Please run 'python main.py --save' first."
        )
    
    model = joblib.load(model_path)
    processor = joblib.load(processor_path)
    
    model_info = {
        "model_name": type(model).__name__,
        "feature_columns": processor.feature_columns,
        "model_type": str(type(model)),
        "loaded": True
    }
    
    return True


def get_confidence_level(probability: float) -> str:
    """Convert probability to confidence level."""
    if probability < 0.3 or probability > 0.7:
        return "high"
    elif probability < 0.4 or probability > 0.6:
        return "medium"
    else:
        return "low"


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        load_models()
        print("Models loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("API will start but predictions won't be available until models are trained.")


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Insurance Enrollment Prediction API is running",
        "models_loaded": model is not None
    }


@app.get("/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using 'python main.py --save'"
        )
    
    return ModelInfoResponse(**model_info)


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(employee: EmployeeData):
    """
    Make a single prediction for an employee.
    
    Returns the predicted enrollment status and probability.
    """
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using 'python main.py --save'"
        )
    
    try:
        # Convert input to dictionary
        data = employee.model_dump()
        
        # Prepare data for prediction
        X = processor.prepare_single_prediction(data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        return PredictionResponse(
            enrolled=int(prediction),
            probability=float(probability),
            confidence=get_confidence_level(probability)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple employees.
    
    Returns a list of predictions with enrollment status and probability.
    """
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using 'python main.py --save'"
        )
    
    if len(request.employees) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 1000 employees per request."
        )
    
    try:
        predictions = []
        
        for employee in request.employees:
            data = employee.model_dump()
            X = processor.prepare_single_prediction(data)
            
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            
            predictions.append(PredictionResponse(
                enrolled=int(prediction),
                probability=float(probability),
                confidence=get_confidence_level(probability)
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload", tags=["Admin"])
async def reload_models():
    """Reload models from disk."""
    try:
        load_models()
        return {"status": "success", "message": "Models reloaded successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
