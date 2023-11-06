# Problem description	
## Stroke Prediction with Machine Learning

### Introduction:

Stroke is a critical and often life-threatening medical condition. Early prediction of stroke risk can aid in timely intervention and potentially save lives. In this project, we will develop a machine learning model to predict the likelihood of an individual experiencing a stroke based on a comprehensive set of health-related attributes and medical history.

### Dataset:

We will be working with a dataset that includes a range of features related to patients' health and lifestyle, as well as their medical history. The dataset contains the following columns:

Age: Age of the patient.
Gender: Gender of the patient (e.g., male, female).
Hypertension: Whether the patient has hypertension (0 - No, 1 - Yes).
Heart Disease: Whether the patient has a history of heart disease (0 - No, 1 - Yes).
Marital Status: Marital status of the patient.
Work Type: The type of work the patient is engaged in.
Residence Type: The type of residence (urban or rural).
Average Glucose Level: The average glucose level in the patient's blood.
Body Mass Index (BMI): The patient's BMI.
Smoking Status: The patient's smoking status (formerly smoked, never smoked, smokes currently).
Alcohol Intake: The patient's alcohol consumption habits.
Physical Activity: The level of physical activity (e.g., sedentary, moderately active, highly active).
Stroke History: Whether the patient has a previous history of stroke (0 - No, 1 - Yes).
Family History of Stroke: Whether there is a family history of stroke (0 - No, 1 - Yes).
Dietary Habits: The patient's dietary preferences and habits.
Stress Levels: The perceived stress levels of the patient.
Blood Pressure Levels: Blood pressure measurements.
Cholesterol Levels: Cholesterol measurements.
Symptoms: Any symptoms or warning signs experienced by the patient.
Diagnosis: The diagnosis related to stroke risk (0 - No stroke, 1 - Stroke).
Objective:

The primary goal of this project is to build a machine learning model capable of predicting whether a patient is at risk of experiencing a stroke based on the provided set of health-related attributes and medical history. This prediction can assist healthcare professionals in identifying high-risk individuals and taking preventive measures.

### Tasks:

Data Preprocessing: Clean the dataset, handle missing values, and encode categorical variables.

Exploratory Data Analysis (EDA): Explore the dataset to gain insights into the distribution of features and their relationships with the target variable.

Feature Selection: Identify the most relevant features for stroke prediction.

Model Building: Train and evaluate machine learning models, such as logistic regression, decision trees, random forests, support vector machines, and gradient boosting, for stroke prediction.

Model Evaluation: Utilize appropriate evaluation metrics, such as accuracy, precision, recall, F1-score, and ROC AUC, to assess model performance.

Hyperparameter Tuning: Optimize the selected model for the best performance.

# Dependency and enviroment management
INSTALL PIPENV DEPENDENCIES FROM PROVIDED PIPFILE
```
pipenv install --system --deploy
```

# Containerization
DOCKER BUILD
```
docker build -t ayarmole_midterm .
```

DOCKER RUN
```
docker run -it --rm -p 9696:9696 ayarmole_midterm
```

RUN REQUEST SCRIPT(outside docker container)
```
python3 flask_requests.py
```
