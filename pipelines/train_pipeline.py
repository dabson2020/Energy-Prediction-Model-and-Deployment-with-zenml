from zenml import pipeline
import pandas as pd
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(data_path: str = "data/CE802_P2_Data.csv"):

    """
    The training pipeline
    Args:
        data_path: path to the data
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test =clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    test_accuracy, test_precision,test_recall,test_f1=evaluate_model(model, X_test, y_test)