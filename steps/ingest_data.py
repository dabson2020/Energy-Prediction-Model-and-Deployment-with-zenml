import logging #(for seeing your log meaage on the terminal)
import pandas as pd
from zenml import step
import pypyodbc as odbc
from steps.connection import username, password

#create a class for ingesting data
class IngestData:
    """
    A class to ingest data from the data path
    """
    def __init__(self, data_path):
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
        server = 'olist-new-server.database.windows.net'
        database = 'energy_data'
        connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password
        conn = odbc.connect(connection_string)

        sql = '''SELECT * from [energy_data]'''

        cursor = conn.cursor()
        cursor.execute(sql)

        # Fetch all the data returned by the query
        data = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(data, columns=columns)  
        return df
    
@step
def ingest_df(data_path):
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