import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging

def load_params(param_path):
    try:
        with open(param_path,"r") as file:
            params = yaml.safe_load(file)
        logging.debug("parameters retrieved from %s", param_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", param_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML Error: %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    
def load_data(data_url):
    try:
        data = pd.read_csv(data_url)
        data = data.iloc[:,1:]
        logging.info("Data loaded from %s: ", data_url)
        return data
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV File: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading file: %s", e)
        raise
    
def preprocess_data(data):
    try:
        logging.info("pre-processing...")
        final_data = data[data['target'].isin(['Positive','Negative','Neutral'])]
        final_data['target'] = final_data['target'].replace({'Positive':1,'Negative':0,'Neutral':2})
        logging.info("Data preprocessing completed")
        return final_data
    except KeyError as e:
        logging.error("Missing column in dataframe: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error while preprocessing data: %s", e)
        raise

def save_data(train_data,test_data,data_path):
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug("train and test data saved to %s", raw_data_path)
    except Exception as e:
        logging.error("Unexpected error while saving data: %s", e)
        raise
    
def main():
    try:
        params = load_params(param_path="params.yaml")
        test_size = params['data_ingestion']['test_size']
        
        # test_size=0.2
        
        data = load_data(data_url="https://raw.githubusercontent.com/gauravbosamiya/Datasets/refs/heads/main/data.csv")
        final_data = preprocess_data(data)
        train_data, test_data = train_test_split(final_data, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")

if __name__ =="__main__":
    main()
        