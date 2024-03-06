import logging

import mlflow
import pandas as pd
from zenml import step
from sklearn.base import ClassifierMixin
from src.evaluation import Accuracy, Precision, Recall, F1_Score
from typing_extensions import Annotated
from typing import Tuple

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   X_test:pd.DataFrame,
                   y_test: pd.Series) -> Tuple[
                       Annotated[float, 'test_accuracy'], 
                       Annotated[float, 'test_precision'], 
                       Annotated[float, 'test_recall'],
                       Annotated[float, 'test_f1'],
                       #Annotated[double, 'test_report'],
                       ]:
    """
    Method to evaluate the model on test data
    Args:
        df: the ingested data

    """


    try:
        
    
        prediction = model.predict(X_test)

        accuracy_class = Accuracy()
        test_accuracy = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("Accuracy", test_accuracy)

        precision_class = Precision()
        test_precision = precision_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("Precision", test_precision)

        recall_class = Recall()
        test_recall = recall_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("Recall", test_recall)

        f1_class = F1_Score()
        test_f1 = f1_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("F1_Score", test_f1)

    

        return test_accuracy, test_precision, test_recall, test_f1
        
    
    except Exception as e:  
        logging.error(f"Error in evaluating model: {e}")
        return e