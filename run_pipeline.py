from zenml.client import Client
from steps.clean_data import clean_df
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model
from steps.connection import username, password

from pipelines.train_pipeline import train_pipeline

server = 'olist-new-server.database.windows.net'
database = 'energy_data'
connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password
df = ingest_df(connection_string)
if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())   
    # Run the pipeline
    train_pipeline(ingest_df=ingest_df, clean_df=clean_df, train_model=train_model, evaluate_model=evaluate_model).run()


