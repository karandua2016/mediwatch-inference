from fastapi import FastAPI
import joblib
import pandas as pd
from typing import List, Union
from contextlib import asynccontextmanager
import os

xgb_model = None

@asynccontextmanager
async def xgboost_model_lifespan(app: FastAPI):
    # Load the ML model

    model_name = os.environ.get('MODEL_NAME') if os.environ.get('MODEL_NAME') else "patient-readmission-prediction"
    
    try:
        global xgb_model 
        xgb_model = joblib.load(f"{model_name}.joblib")
        print("Model loaded from MLflow Server")
        yield
    except Exception:
        print("Model could not be loaded. Exiting")
        exit()

    # Clean up the ML models and release the resources
    xgb_model = None

app = FastAPI(lifespan=xgboost_model_lifespan)

# Define the prediction endpoint
@app.post("/predict")
async def predict(features: List[List[Union[float, int]]]):

    # Convert the input features to a DataFrame
    feature_columns = [f"feature_{i}" for i in range(len(features[0]))]
    input_data = pd.DataFrame(features, columns=feature_columns)

    # Make predictions
    predictions = xgb_model.predict(input_data)

    return {"predictions": predictions.tolist()}
