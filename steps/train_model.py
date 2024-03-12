import logging

import mlflow
import pandas as pd
from zenml import step

from src.model_dev import LogisticRegression, DecisionTree, RandomForest, GradientBoost, XGB
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series ,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> ClassifierMixin:
    """
    Method to train the model on the training data

    Args:
        X_train: training data
        X_test: testing data
        y_train: training labels
        y_test: testing labels
    Returns:
        trained model
    """
    try:
        model = None
        if config.model_name == "LogisticRegression": 
            mlflow.sklearn.autolog() 
            model = LogisticRegression()
            lg_model = model.train(X_train, y_train)
            return lg_model
        elif config.model_name == "DecisionTree":
            mlflow.sklearn.autolog(disable=False) 
            model = DecisionTree()
            dt_model = model.train(X_train, y_train)
            return dt_model
        elif config.model_name == "RandomForest":
            mlflow.sklearn.autolog(disable=False) 
            model = RandomForest()
            rf_model = model.train(X_train, y_train)
            return rf_model
        elif config.model_name == "GradientBoost":
            mlflow.sklearn.autolog(disable=False) 
            model = GradientBoost()
            gb_model = model.train(X_train, y_train)
            return gb_model     
        elif config.model_name == "XGB":
            mlflow.sklearn.autolog(disable=False) 
            #mlflow.log_model(model, "model")
            model = XGB()
            xgb_model = model.train(X_train, y_train)
            return xgb_model
            
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        return e
   