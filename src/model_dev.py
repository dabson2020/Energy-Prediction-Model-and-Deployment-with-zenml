import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

class Model(ABC):
    """
    Abstract class for all model
    """
    @abstractmethod
    def train(self, X_train,y_train) ->None:
        """
        Method to train the model

        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
       """
        
class LogisticRegression(Model):
    """
    Class for Logistic Regression model
    """
    
    def train(self, X_train, y_train, **kwargs): 

        """
        Method to train the model
        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        try:
            lg = LogisticRegression(**kwargs) 
            lg.fit(X_train, y_train)
            logging.info(f"Logistic Regression Model is trained successfully")   
            return lg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            return e

class DecisionTree(Model):
    """
    Class for Decision Tree model
    """
    
    def train(self, X_train, y_train, **kwargs): 

        """
        Method to train the model
        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        try:
            dt = DecisionTreeClassifier(**kwargs) 
            dt.fit(X_train, y_train)
            logging.info(f"Decision Tree Model is trained successfully")   
            return dt
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            return e

class RandomForest(Model):
    """
    Class for Logistic Regression model
    """
    
    def train(self, X_train, y_train, **kwargs): 

        """
        Method to train the model
        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        try:
            rf = RandomForestClassifier(**kwargs) 
            rf.fit(X_train, y_train)
            logging.info(f"Random Forest Model is trained successfully")   
            return rf
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            return e

class GradientBoost(Model):
    """
    Class for Logistic Regression model
    """
    
    def train(self, X_train, y_train, **kwargs): 

        """
        Method to train the model
        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        try:
            gb = GradientBoostingClassifier(**kwargs) 
            gb.fit(X_train, y_train)
            logging.info(f"Gradient Boost Model is trained successfully")   
            return gb
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            return e

class XGB(Model):
    """
    Class for Logistic Regression model
    """
    
    def train(self, X_train, y_train, **kwargs): 

        """
        Method to train the model
        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        try:
            xgb = XGBClassifier(**kwargs) 
            xgb.fit(X_train, y_train)
            logging.info(f"XGB Model is trained successfully")   
            return xgb
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            return e