import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy

def get_data_for_test():
    """
    Method to get the data for testing
    Returns:
        pandas dataframe
    """
    try:
        data = pd.read_csv("./data/CE802_P2_Test.csv")
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy) 
        data = data_cleaning.handle_data()
        data = data.drop("Class", axis=1)
        result = data.to_json(orient="split")
        return result

    except Exception as e:
        logging.error(e)
        raise e