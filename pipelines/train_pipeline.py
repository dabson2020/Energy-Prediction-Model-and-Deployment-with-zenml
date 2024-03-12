from zenml import pipeline
import mlflow
import pandas as pd
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from zenml.pipelines import pipeline
from steps.connection import username, password

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(ingest_df=ingest_df, clean_df=clean_df, train_model=train_model, evaluate_model=evaluate_model):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        accuracy: float
        precision: float
        recall: float
        f1: float
    
    """
    server = 'olist-new-server.database.windows.net'
    database = 'energy_data'
    connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password

    df = ingest_df(connection_string)
    X_train, X_test, y_train, y_test =clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    #mlflow.sklearn.log_model(model, "model")
    test_accuracy, test_precision,test_recall,test_f1=evaluate_model(model, X_test, y_test)
    



        