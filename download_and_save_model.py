import os
import mlflow
import joblib

def download_from_mlflow_and_save():

    model_name = os.environ.get('MODEL_NAME') if os.environ.get('MODEL_NAME') else "mediwatch-prediction"

    xgb_model = mlflow.xgboost.load_model(f"models:/{model_name}@production")
    
    # Below is deprecated way of loading the model. See https://mlflow.org/docs/2.9.2/model-registry.html#migrating-from-stages
    # xgb_model = mlflow.xgboost.load_model(f"models:/{model_name}/production")

    print("Model loaded from MLflow Server and Saved to current directory")
    joblib.dump(xgb_model, f'{model_name}.joblib')

if __name__ == '__main__':
    download_from_mlflow_and_save()