import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deploy_pipeline import prediction_service_loader
from run_deployment import run_deployment_pipeline

def main():
    st.title("Energy Consumption Prediction")

    st.write("This is a simple web app to predict if a consumer will be able to pay the rising energy consumption.")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict if a consumer will be able to pay the rising energy consumption.
   """)
    st.markdown(
        """ 
        Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and re-trains the model. If the model meets minimum accuracy requirement, the model will be deployed.
        """
    )

    
    
    # Define the data columns with their respective values
    
    F1 = st.number_input("F1: ")
    F2 = st.number_input("F2: ")
    F3 = st.number_input("F3: ")
    F4 = st.number_input("F4: ")
    F5 = st.number_input("F5: ")
    F6 = st.number_input("F6: ")
    F7 = st.number_input("F7: ")
    F8 = st.number_input("F8: ")
    F9 = st.number_input("F9: ")
    F10 = st.number_input("F10: ")
    F11 = st.number_input("F11: ")
    F12 = st.number_input("F12: ")
    F13 = st.number_input("F13: ")
    F14 = st.number_input("F14: ")
    F15 = st.number_input("F15: ")
    F16 = st.number_input("F16: ")
    F17 = st.number_input("F17: ")
    F18 = st.number_input("F18: ")
    F19 = st.number_input("F19: ")
    F20 = st.number_input("F20: ")
    F21 = st.number_input("F21: ")

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_deployment_pipeline()
        try:
            data_point = {
            'F1': float(F1),
            'F2': float(F2),
            'F3': float(F3),
            'F4': float(F4),
            'F5': float(F5),
            'F6': float(F6),
            'F7': float(F7),
            'F8': float(F8),
            'F9': float(F9),
            'F10': float(F10),
            'F11': float(F11),
            'F12': float(F12),
            'F13': float(F13),
            'F14': float(F14),
            'F15': float(F15),
            'F16': float(F16),
            'F17': float(F17),
            'F18': float(F18),
            'F19': float(F19),
            'F20': float(F20),
            'F21': float(F21),
            }

            # Convert the data point to a Series and then to a DataFrame
            data_point_series = pd.Series(data_point)
            data_point_df = pd.DataFrame(data_point_series).T

            # Convert the DataFrame to a JSON list
            json_list = json.loads(data_point_df.to_json(orient="records"))
            data = np.array(json_list)
            for i in range(len(data)):
                logging.info(data[i])
            pred = service.predict(data)
            logging.info(pred)
            st.success(f"Customer churn prediction: {'Churn' if pred == 1 else 'No Churn'}")
        except Exception as e:
            logging.error(e)
            raise e

        
if __name__ == "__main__":
    main()