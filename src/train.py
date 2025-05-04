import mlflow.sklearn
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

os.environ['MLFLOW_TRCKING_URI'] = "YOUR DAGSHUB REPOSITORY LINK"
os.environ['MLFLOW_TRACKING_USERNAME'] = "YOUR DAGSHUB USERNAME"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "YOUR DAGSHUB TOKEN"


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

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRCKING_URI'])

    ## Start the MLFlow run
    with mlflow.start_run():
        ## Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        signature = infer_signature(X_train, y_train)

        ## Define the Hyperparameters grid

        params_grid = {
            'n_estimators' : [100, 200],
            'max_depth' : [5, 10, None],
            'min_samples_split' : [2, 5],
            'min_samples_leaf' : [1, 2],
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, params_grid)

        ## Get the best model and parameters
        best_model = grid_search.best_estimator_

        ## Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        ## Log additional metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_param('best_samples_split', grid_search.best_params_['min_samples_split'])
        mlflow.log_param('best_samples_leaf', grid_search.best_params_['min_samples_leaf'])

        ## Log the confusion matrix and classification report
        cm = confusion_matrix(y_test, y_test)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "Confusion_Matrix.txt")
        mlflow.log_text(cr, "Classification_Report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="RandomForestClassifier")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        ## Create the directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True) 

        file_name = model_path
        pickle.dump(best_model, open(file_name, 'wb'))

        print(f"Model saved to {file_name}")


if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])




        


