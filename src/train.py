import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRCKING_URI'] = "https://dagshub.com/forcoding247/Machine-Learning-PipeLine.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "forcoding247"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "f8a0e72116c9a3692360c53afb2c4baaed03b176"


def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']


def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/forcoding247/Machine-Learning-PipeLine.mlflow")
