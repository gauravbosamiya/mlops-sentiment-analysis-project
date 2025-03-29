import numpy as np 
import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
from src.logger import logging

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded from %s:",file_path)
        return data
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV File: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading file: %s", e)
        raise
    
def train_model(X_train,y_train):
    try:
        rfc = RandomForestClassifier(max_depth=None, min_samples_split=4, n_estimators=100, random_state=42)
        rfc.fit(X_train,y_train)
        logging.info("Model training completed..")
        return rfc
    except Exception as e:
        logging.error("Error during model training: %s",e)
        raise
    
def save_model(model,file_path):
    try:
        with open(file_path,"wb") as file:
            pickle.dump(model,file)
        logging.info("Model saved to: %s", file_path)
    except Exception as e:
        logging.error("Error occured while saving the model: %s",e)
        raise
    
def main():
    try:
        train_data = load_data("./data/processed/train_bow.csv")
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        rfc = train_model(X_train,y_train)
        save_model(rfc,"models/model.pkl")
    except Exception as e:
        logging.error("Failed to complete the model building process: %s",e)
        print(f"Error:{e}")
        
if __name__=="__main__":
    main()