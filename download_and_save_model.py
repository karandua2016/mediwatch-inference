import os
import mlflow
import joblib
import cloudpickle
import sys

common_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common_config'))
sys.path.append(common_config_path)

import common_config as cmnfg

def download_from_mlflow_and_save():

    model_name = os.environ.get('MODEL_NAME') if os.environ.get('MODEL_NAME') else "mediwatch-prediction"

    client = mlflow.client.MlflowClient()

    # Downloading the transformation pipeline object for packaging inside docker image
    mv = client.get_model_version_by_alias(model_name, alias="production")

    mlflow.artifacts.download_artifacts(mv.source.replace('model', cmnfg.transformation_pipeline_file_name), dst_path='.')

    xgb_model = mlflow.xgboost.load_model(f"models:/{model_name}@production")

    # run_id = mlflow.get_run(xgb_model._meta.mlflow_env['run_id']).info.run_id
    
    # Below is deprecated way of loading the model. See https://mlflow.org/docs/2.9.2/model-registry.html#migrating-from-stages
    # xgb_model = mlflow.xgboost.load_model(f"models:/{model_name}/production")

    print("Model and transformation pipeline loaded from MLflow Server and Saved to current directory")
    joblib.dump(xgb_model, f'{model_name}.joblib')

if __name__ == '__main__':
    download_from_mlflow_and_save()