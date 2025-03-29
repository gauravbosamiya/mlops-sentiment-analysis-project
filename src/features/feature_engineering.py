import numpy as np 
import pandas as pd 
import os 
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle

def load_params(params_path):
    try:
        with open(params_path,"r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML Error: %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.fillna('',inplace=True)
        logging.info("Data loaded and None values are filled from %s", file_path)
        return data
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV File: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading file: %s", e)
        raise

def apply_bow(train_data,test_data,max_features):
    try:
        logging.info("Applying bow...")
        vectorizer = CountVectorizer(max_features=max_features)
        
        X_train = train_data['content'].values
        y_train = train_data['target'].values
        X_test = test_data['content'].values
        y_test = test_data['target'].values
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_data = pd.DataFrame(X_train_bow.toarray())
        train_data['label'] = y_train
        
        test_data = pd.DataFrame(X_test_bow.toarray())
        test_data['label'] = y_test
        
        pickle.dump(vectorizer, open('models/vectorizer.pkl','wb'))
        logging.info("Bag of words applied and data transformed")
        
        return train_data, test_data
    except Exception as e:
        logging.error("Error during bag of words transformation: %s", e)
        raise
    
def save_data(data,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        data.to_csv(file_path,index=False)
        logging.info("Data saved to %s", file_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data: %s", e)
        raise
    
    
def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        # max_features = 100
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        train_df, test_df = apply_bow(train_data, test_data, max_features)
        save_data(train_df, os.path.join("./data","processed","train_bow.csv"))
        save_data(test_df, os.path.join("./data","processed","test_bow.csv"))
    except Exception as e:
        logging.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error:{e}")
        
if __name__=="__main__":
    main()