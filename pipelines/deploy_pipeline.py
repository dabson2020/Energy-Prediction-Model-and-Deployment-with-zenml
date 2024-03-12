import numpy as np
import pandas as pd
import logging
import json
import pypyodbc as odbc
#from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_df
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model
from steps.connection import username, password


import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from .util import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger configuration"""
    min_accuracy: float = 0.84

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Dynamic importer step to import the data from the data path"""
    data = get_data_for_test()
    return data

    
    
@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
   
):
    """Implements a simple model that looks at the input model 
    accuracy and determines whether the model should be deployed or not."""
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    df = json.loads(data)
    df.pop("columns")
    df.pop("index")
    columns_for_df = ['F1','F2','F3','F4','F5','F6','F7','F8','F9',
                      'F10','F11','F12','F13','F14','F15','F16',
                      'F17','F18','F19','F20','F21']
       
    df = pd.DataFrame(df["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    df = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.84,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    server = 'olist-new-server.database.windows.net'
    database = 'energy_data'
    connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password

    data = ingest_df(connection_string)
    X_train, X_test, y_train, y_test =clean_df(data)
    model = train_model(X_train, X_test, y_train, y_test)
    test_accuracy, test_precision,test_recall,test_f1=evaluate_model(model, X_test, y_test)

    deployer_decision = deployment_trigger(test_accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployer_decision,
        workers=workers,
        timeout=timeout,

    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=model_deployment_service, data=batch_data)
    return prediction
