import logging #(for seeing your log meaage on the terminal)
import pandas as pd
from zenml import step

#create a class for ingesting data
class IngestData:
    """
    A class to ingest data from the data path
    """
    def __init__(self, data_path: str):
        """
        Constructor to initialize the data path
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Method to ingest the data from the data_path
        Returns:
            pandas dataframe
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv("data/CE802_P2_Data.csv")
    
@step
def ingest_df(data_path: str):
    """
    ingesting data from the data path

    Args:
        data_path: path to the data.
    Returns:
        pandas dataframe
    """
    try:
        ingest_data=IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in ingesting data: {e}")
        return e