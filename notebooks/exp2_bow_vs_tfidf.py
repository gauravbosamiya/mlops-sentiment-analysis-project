import setuptools
import os
import pandas as pd 
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "data_path" : "notebooks/data.csv",
    "test_size" : 0.2,
    "mlflow_tracking_uri" :  "https://dagshub.com/gauravbosamiya/mlops-sentiment-analysis-project.mlflow",
    "dagshub_repo_owner" : "gauravbosamiya",
    "dagshub_repo_name" : "mlops-sentiment-analysis-project",
    "experiment_name" : "EXP2_BOW vs TFIDF"
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"],repo_name=CONFIG["dagshub_repo_name"],mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])


def lemmatization(text):
    lemmitizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmitizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = "".join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  
    text = re.sub('\s+', ' ', text).strip()  
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(data):
    try:
        data['content'] = data['content'].apply(lower_case)
        data['content'] = data['content'].apply(remove_stop_words)
        data['content'] = data['content'].apply(removing_numbers)
        data['content'] = data['content'].apply(removing_punctuations)
        data['content'] = data['content'].apply(removing_urls)
        data['content'] = data['content'].apply(lemmatization)
        return data
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data = normalize_text(data)
        data = data[data['target'].isin(['Positive','Negative','Neutral'])]
        data['target'] = data['target'].replace({'Positive':1,'Negative':0,'Neutral':2})
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    
VECTORIZER = {
    'bow' : CountVectorizer(),
    'TF-IDF' : TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression' : LogisticRegression(),
    'MultinomialNB' : MultinomialNB(),
    'RandomForest' : RandomForestClassifier(),
    'GradientBoosting' : GradientBoostingClassifier()
}

def train_and_evaluate(data):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algo in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZER.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}",nested=True) as child_run:
                    try:
                        X = vectorizer.fit_transform(data['content'])
                        y = data['target']
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=CONFIG["test_size"], random_state=42)
                        
                        
                        mlflow.log_params({
                            "vectorizer" : vec_name,
                            "algorithm" : algo_name,
                            "test_size" : CONFIG["test_size"]
                        })
                        
                        model = algo
                        model.fit(X_train,y_train)
                        
                        log_model_params(algo, model)
                        
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy" : accuracy_score(y_test,y_pred),
                            "precision" : precision_score(y_test,y_pred,average="macro"),
                            "recall" : recall_score(y_test,y_pred,average="macro"),
                            "f1" : f1_score(y_test,y_pred,average="macro")
                        }
                        
                        mlflow.log_metrics(metrics)
                        
                        mlflow.sklearn.log_model(model, "model")
                        
                        print(f"\n Algorithm : {algo_name}, Vectorizer :{vec_name}")
                        print(f"Metrics: {metrics}")
                        
                    except Exception as e:
                       print(f"Error in training: {algo_name} with {vec_name}: {e}")
                       mlflow.log_param("error",str(e))
                       
def log_model_params(algo_name, model):
    params_to_log = {}
    if algo_name=="LogisticRegression":
        params_to_log["C"] = model.C 
    elif algo_name=="MultinomialNB":
        params_to_log["alpha"] = model.alpha
    elif algo_name=="RandomForest":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name=="GradientBoosting":
        params_to_log["n_estimator"] = model.n_estimators 
        params_to_log["max_depth"] = model.max_depth
        params_to_log["learning_rate"] = model.learning_rate 
        
    mlflow.log_params(params_to_log)
    
if __name__ == "__main__":
    data = load_data(CONFIG["data_path"])
    train_and_evaluate(data)
                    
