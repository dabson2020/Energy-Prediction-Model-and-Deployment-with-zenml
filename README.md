## 1. Energy Prices Prediction in the United Kingdom:

Due to inflation, the energy price has increased in the United Kingdom. The government wants to determine the families and individuals who cannot meet this price increase. There are many factors (features) here to determine or predict the inability to meet up with energy prices. These features are categorized into four (4):
  - Temperature: The weather of a day could have a significant effect on the energy consumption by consumers. In winter, the temperature is cold, and to keep warm, energy is consumed through the heating system. During summer, the temperature rises and as such, the use of the heating system is at its minimum or not in use at all. But due to high temperatures, there is a need to drop the temperature by air-conditioning. Both situations may lead to an increase in energy consumption.
 - Appliances: Another category that influences energy consumption is appliances. The features to consider here may include the type of appliances such as Televisions, washing machines, dryers, microwaves, and electric kettles. Another feature here is the number of each appliance and the frequency of their uses. Other features may include age and the appliances’ energy rating, which can significantly affect electricity consumption. The lower the age of an appliance, the less consumption of energy. Due to technological advancements, newer appliances tend to consume less energy. The lower number of electrical appliances and decreases in usage may reduce energy consumption and vice versa, influencing the increase or decrease of electricity cost.
- Consumer habit: This feature can either increase or decrease users' electricity consumption. Little changes made to these features which may include ensuring electrical appliances are switched off when not used, turning off light bulbs and heaters, reducing the temperature of the water heater, and switching off all devices before sleeping or when not in the home can help reduce consumer’s electricity consumption and as a result decrease in electricity cost. The vice versa can lead to an increase in electricity cost.
- Demography: There are quite a few features to consider here. The characteristics of the consumer which may include age, gender, income, education, home ownership, sexual orientation, marital status, family size, health, and disability status can influence energy consumption and the ability to pay increasing electricity costs. If the family size is large, there is a tendency for an increase in electricity consumption and an overall increase in electricity cost. With a high level of education and good income, the obligation to pay the increasing electricity cost may not be difficult for the consume
About 26 features are categorized into the 4 groups above, which are explored and analyzed.

The project is divided into two:

**Classification Problem**: With these features, we are to determine if a consumer is having difficulties with increasing energy prices or not. The target variable and prediction is the True (1) value if the customer is having difficulties with the increasing energy prices and False (0) if vice versa.

The following processes were considered:

- **Data ingestion/Loading of Data**
- **Data Cleaning and preprocessing**
- **Feature Engineering**
- **Model Development and Prediction**
  
  - The data was split into train and test set
  - The whole data was used for cross-validation with CV = 5. This means that the model will split the data into 5, using 4 folds of the data as training data to train the model and 1 fold as test data to test the model. This will be done five times with different folds as test data. At any point, the test data are different. They help to validate the model and see if the model is generalized and not overfitting or underfitting when unknown (unseen data) is used to test the model.
  - Nine (9) models were developed which include LGBM, XGB, GradientBoost, Random Forest, Quadratic Discriminant, Linear Discriminant, Logistic Regression, and GaussianNB. 
  - The model performance was determined with the following metrics:
    - Accuracy
    - Precision
    - Recall
    - Confusion Matrix
    - Classification Report

  The table below displays the performance of each trained model on test data.
  
  ![image](https://github.com/dabson2020/Projects-Portfolio/assets/45830157/d1792e76-7645-4b35-a0a9-7a5dade84075)
  
  Here is a visual representation of the model performance
  
  ![image](https://github.com/dabson2020/Projects-Portfolio/assets/45830157/d657646c-9e04-4ac9-b390-b2f8cb11304d)


  With the accuracy of 89%, 89% on cross-validation score on test data, and 90.1% accuracy on unknown data, the model with the best performance is the **XGB Classifier**

  The model was deployed and predicted with Zenml.
  
