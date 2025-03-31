import numpy as np 
import pandas as pd 
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os 
from src.logger import logging

# production use
# ------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner="gauravbosamiya"
repo_name="mlops-sentiment-analysis-project"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# -------------------------------------------------------------------------------
# for local use
# mlflow.set_tracking_uri("https://dagshub.com/gauravbosamiya/mlops-sentiment-analysis-project.mlflow")
# dagshub.init(repo_owner="gauravbosamiya",repo_name="mlops-sentiment-analysis-project",mlflow=True)

def load_model(file_path):
    try:
        with open(file_path,"rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading model: %s", e)
        raise
    
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded from: %s",file_path)
        return data
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV File: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading file: %s", e)
        raise
    
def evaluate_model(rfc,X_test,y_test):
    try:
        y_pred = rfc.predict(X_test)
        y_pred_proba = rfc.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average="macro")
        recall = recall_score(y_test,y_pred,average="macro")
        
        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
        }  
        logging.info("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logging.error("Error during model evaluation: %s",e)
        raise
    
def save_metrics(metrics,file_path):
    try:
        with open(file_path,"w") as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved to: %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the metrics: %s",e)
        raise
    
    
    
def save_model_info(run_id, model_path, file_path):
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            rfc = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(rfc, X_test, y_test)

            if not isinstance(metrics, dict):
                raise TypeError(f"Expected dictionary but got {type(metrics)}: {metrics}")

            save_metrics(metrics, 'reports/metrics.json')

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            if hasattr(rfc, 'get_params'):
                params = rfc.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)  

            mlflow.sklearn.log_model(rfc, "model")

            save_model_info(run.info.run_id, "model", "reports/experiment_info.json")  

            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error("Failed to complete the model evaluation process: %s", e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
            