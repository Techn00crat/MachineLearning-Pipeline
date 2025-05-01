import pandas as pd
import mlflow
import pickle
import os
import yaml
from sklearn.metrics import accuracy_score

from urllib.parse import urlparse

os.environ['MLFLOW_TRCKING_URI'] = "https://dagshub.com/forcoding247/Machine-Learning-PipeLine.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "forcoding247"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "f8a0e72116c9a3692360c53afb2c4baaed03b176"


## Loading the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)

    X= data.drop(columns=['Outcome'])
    y= data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRCKING_URI'])


    ## Loading the model
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    ## Log the metrics to MLflow

    mlflow.log_metric("Accuracy", accuracy)
    print(f"Model Accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate(params['data'], params['model'])