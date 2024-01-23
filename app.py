from fastapi import FastAPI
import joblib
import pandas as pd
from typing import List, Union
from contextlib import asynccontextmanager
import os
import cloudpickle
from sklearn import set_config

xgb_model = None
pipe = None

@asynccontextmanager
async def xgboost_model_lifespan(app: FastAPI):
    # Load the ML model

    model_name = os.environ.get('MODEL_NAME') if os.environ.get('MODEL_NAME') else "mediwatch-prediction"
    
    try:
        global xgb_model 
        global pipe

        xgb_model = joblib.load(f"{model_name}.joblib")

        with open('pipeline.pkl', 'rb') as file:
            pipe = cloudpickle.load(file)

        print("Model and tranformer object loaded")
        yield
    except Exception:
        print("Model could not be loaded. Exiting")
        exit()

    # Clean up the ML models and release the resources
    xgb_model = None
    pipe = None

app = FastAPI(lifespan=xgboost_model_lifespan)

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: dict):

    set_config(transform_output="pandas")

    input_data = pd.DataFrame.from_dict(data, orient='index').T

    input_data = pipe.transform(input_data)

    if 'readmitted' in input_data.columns:
        input_data = input_data.drop(columns=['readmitted'])

    # Make predictions
    predictions = xgb_model.predict(input_data.values)

    return {"predictions": predictions.tolist()}
