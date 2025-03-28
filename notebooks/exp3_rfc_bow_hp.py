import setuptools
import os
import pandas as pd 
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")


mlflow.set_tracking_uri("https://dagshub.com/gauravbosamiya/mlops-sentiment-analysis-project.mlflow")
dagshub.init(repo_owner="gauravbosamiya",repo_name="mlops-sentiment-analysis-project",mlflow=True)

mlflow.set_experiment("Rfc Hyperparameter Tuning")



def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = text.lower()
    text = re.sub(r'\d+','',text)
    text = re.sub(f"[{re.escape(string.punctuation)}]"," ",text)
    text  = re.sub(r'https?://\S+|www\.\S+','',text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    
    return text.strip()


def load_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data['content'] = data['content'].astype(str).apply(preprocess_text)
    
    data = data[data['target'].isin(['Positive','Negative','Neutral'])]
    data['target'] = data['target'].replace({'Positive':1,'Negative':0,'Neutral':2})
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data["content"])
    y = data["target"]
    
    return train_test_split(X,y,test_size=0.2,random_state=42), vectorizer

def train_and_log_model(X_train,X_test,y_train,y_test,vectorizer):
    param_grid = {
        "n_estimators" : [100,150],
        "max_depth" : [None,5,10],
        "min_samples_split" : [2,4],
    }
    with mlflow.start_run():
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train,y_train)
        
        for params,mean_score, std_score in zip(grid_search.cv_results_["params"],
                                                grid_search.cv_results_["mean_test_score"],
                                                grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"RFC with params: {params}", nested=True):
                model = RandomForestClassifier(**params)
                model.fit(X_train,y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy" : accuracy_score(y_test,y_pred),
                    "precision" : precision_score(y_test,y_pred,average="macro"),
                    "recall" : recall_score(y_test,y_pred,average="macro"),
                    "f1" : f1_score(y_test,y_pred,average="macro"),
                    "mean_cv_score" : mean_score,
                    "std_cv_score" : std_score
                } 
                
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Parms: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
                
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_accuracy = grid_search.best_score_
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score",best_accuracy)
        mlflow.sklearn.log_model(best_model,"model")
        
        print(f"\Best params : {best_params} | Best Accuracy : {best_accuracy:.4f}")
        
        
if __name__ =="__main__":
    (X_train,y_train,X_test,y_test), vectorizer = load_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train,y_train,X_test,y_test,vectorizer)
    
    
#  {'max_depth': None, 'min_samples_split': 4, 'n_estimators': 150} | Best Accuracy : 0.9203                
                