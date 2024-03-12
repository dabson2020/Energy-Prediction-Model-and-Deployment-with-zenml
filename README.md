## 1. Energy Prices Prediction in the United Kingdom
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

Due to inflation, the energy price has increased in the United Kingdom. The government wants to determine the families and individuals who cannot meet this price increase. There are many factors (features) here to determine or predict the inability to meet up with energy prices. These features are categorized into four (4):
  - Temperature: The weather of a day could have a significant effect on the energy consumption by consumers. In winter, the temperature is cold, and to keep warm, energy is consumed through the heating system. During summer, the temperature rises and as such, the use of the heating system is at its minimum or not in use at all. But due to high temperatures, there is a need to drop the temperature by air-conditioning. Both situations may lead to an increase in energy consumption.
 - Appliances: Another category that influences energy consumption is appliances. The features to consider here may include the type of appliances such as Televisions, washing machines, dryers, microwaves, and electric kettles. Another feature here is the number of each appliance and the frequency of their uses. Other features may include age and the appliances’ energy rating, which can significantly affect electricity consumption. The lower the age of an appliance, the less consumption of energy. Due to technological advancements, newer appliances tend to consume less energy. The lower number of electrical appliances and decreases in usage may reduce energy consumption and vice versa, influencing the increase or decrease of electricity cost.
- Consumer habit: This feature can either increase or decrease users' electricity consumption. Little changes made to these features which may include ensuring electrical appliances are switched off when not used, turning off light bulbs and heaters, reducing the temperature of the water heater, and switching off all devices before sleeping or when not in the home can help reduce consumer’s electricity consumption and as a result decrease in electricity cost. The vice versa can lead to an increase in electricity cost.
- Demography: There are quite a few features to consider here. The characteristics of the consumer which may include age, gender, income, education, home ownership, sexual orientation, marital status, family size, health, and disability status can influence energy consumption and the ability to pay increasing electricity costs. If the family size is large, there is a tendency for an increase in electricity consumption and an overall increase in electricity cost. With a high level of education and good income, the obligation to pay the increasing electricity cost may not be difficult for the consume
About 26 features are categorized into the 4 groups above, which are explored and analyzed.

The project is divided into two:

**Classification Problem**: With these features, we are to determine if a consumer is having difficulties with increasing energy prices or not. The target variable and prediction is the True (1) value if the customer is having difficulties with the increasing energy prices and False (0) if vice versa.

The following processes were considered:

- **Data ingestion/Loading of Data:** Data is ingested from Azure SQL Database (ingest_data,py)
- **Data Cleaning and preprocessing:** Data is cleaned and preprocessed (clean_data.py)
- **Model Development** XGB model is developed with 3 other models (train_model.py)
- **Model Evaluation** Model is evaluated with accuracy, precision, recall, f1_score (evaluate_model.py)
- **Model Deployment and Prediction** Model is deployed (run_deployment.py)
- **Web application**

The model building, deployment, and prediction are done with Zenml

## Required modules and libraries
pip install -r requirements.txt
**reference to zenml projects:**  git clone https://github.com/zenml-io/zenml-projects.git
pip install zenml['server']
zenml init
zenml up
**NOTE**
Before running the `run_deployment.py` script, install the following:

zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component.

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

To run the run_pipeline.py use:
python run_pipeline.py

To run run_deployment.py for deployment
python run_deployment --config deploy

To run the run_deployment.py for prediction
python run_deployment --config predict

The XGB Model is deployed with an accuracy of 0.905

  
