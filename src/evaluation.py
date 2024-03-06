import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            float
        """
        pass

        
class Accuracy(Evaluation):
    """
    Evaluation Strategy that uses accuracy score to evaluate the model
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            float
        """
        try:
            logging.info(f"Calculating accuracy score of the model")
            acc = accuracy_score(y_true, y_pred)
            logging.info(f"Model Accuracy: {acc}")
            return acc
        except Exception as e:
            logging.error(f"Error in calculating accuracy: {e}")
            return e

class Precision(Evaluation):
    """
    Evaluation Strategy that uses precision score to evaluate the model
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores (precision) for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            f
        """
        try:
            logging.info(f"Calculating Precision Score")
            precision = precision_score(y_true, y_pred)
            logging.info(f"Precision Score: {precision}")
            return precision
        except Exception as e:
            logging.error(f"Error in calculating precision Score: {e}")
            return e  

class Recall(Evaluation):
    """
    Evaluation Strategy that uses Roecall score to evaluate the model
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            np.array
        """
        try:
            logging.info(f"Calculating Recall score")
            recall = recall_score(y_true, y_pred)
            logging.info(f"Recall: {recall}")
            return recall
        except Exception as e:
            logging.error(f"Error in calculating Recall: {e}")
            return e 

class F1_Score(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            np.array
        """
        try:
            logging.info(f"Calculating F1 Score")
            f1 = f1_score(y_true, y_pred)
            logging.info(f"F1 Score: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Error in calculating F1 Score: {e}")
            return e 

class class_report(Evaluation):
    """
    Evaluation Strategy that uses the classification report to evaluate the model
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            np.array
        """
        try:
            logging.info(f"Calculating the classification report")
            report = classification_report(y_true, y_pred)
            logging.info(f"Classification Report: {report}")
            return report
        except Exception as e:
            logging.error(f"Error in computing the classification Report: {e}")
            return e 