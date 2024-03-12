import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy
import pypyodbc as odbc
from steps.connection import username, password

def get_data_for_test():
    """
    Method to get the data for testing
    Returns:
        pandas dataframe
    """
    try:
        server = 'olist-new-server.database.windows.net'
        database = 'energy_data'
        connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password
        conn = odbc.connect(connection_string)

        sql = '''SELECT * from [test_energy_data]'''

        cursor = conn.cursor()
        cursor.execute(sql)

        # Fetch all the data returned by the query
        data = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        data = pd.DataFrame.from_records(data, columns=columns)  
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy) 
        data = data_cleaning.handle_data()
        data = data.drop("class", axis=1)
        df = data.to_json(orient="split")
        return df

    except Exception as e:
        logging.error(e)
        raise e
