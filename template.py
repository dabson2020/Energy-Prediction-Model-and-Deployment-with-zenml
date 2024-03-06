import os
from pathlib import Path
import logging

#this configuration will help initialise a logging string because we want to see 
#the log in our terminal. It will set the timestamp, the time we executed the code
# and execution message or error message.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

#Here we give the project name


#To create the list of files

list_of_files = [
    f'pipelines/train_pipeline.py',
    f'pipelines/deploy_pipeline.py',
    f"src/model_dev.py",
    f"src/data_cleaning.py",
    f"src/evaluation.py",
    "saved_model/model.pkl",
    f'steps/ingest_data.py',
    f'steps/clean_data.py',
    f'steps/evaluate_model.py',
    f'steps/train_model.py',
    f"steps/config.py",
    "__init__.py",
    "config.yaml",
    "requirements.txt",
    "run_deployment.py",
    "run_pipeline.py",
    "streamlit_app.py"

]

#To create the folder and files above, i use a for loop

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")