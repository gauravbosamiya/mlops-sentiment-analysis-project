from src.logger import logging
import os 
import dagshub
import json 
import mlflow
import logging

import warnings
warnings.filterwarnings("ignore")

# production use
# ------------------------------------------------------------------------------
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner="gauravbosamiya"
# repo_name="mlops-sentiment-analysis-project"

# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# -------------------------------------------------------------------------------
# for local use
mlflow.set_tracking_uri("https://dagshub.com/gauravbosamiya/mlops-sentiment-analysis-project.mlflow")
dagshub.init(repo_owner="gauravbosamiya",repo_name="mlops-sentiment-analysis-project",mlflow=True)


def load_model_info(file_path):
    try:
        with open(file_path,"rb") as file:
            model_info = json.load(file)
        logging.debug("model info loaded from: %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("file not found: %s",file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading the model info: %s", e)
        raise
    
def register_model(model_name, model_info):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        model_version = mlflow.register_model(model_uri, model_name)
        
        clinet = mlflow.tracking.MlflowClient()
        clinet.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="staging"
        )
        logging.debug(f"model {model_name} version {model_version.version} registered and transitioned to Stagging.")
    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise
    
def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)
        model_name = "my_model"
        
        
        register_model(model_name, model_info)
        
    except Exception as e:
        logging.error("Failed to complete the model registration process: %s",e)
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()