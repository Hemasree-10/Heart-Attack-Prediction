<h1> Heart Attack Prediction using Streamlit App </h1>

![Streamlit](https://img.shields.io/badge/Deployed%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue?logo=numpy&logoColor=white)

# Project Descriptions
<p align="justify"> The Heart Attack Prediction App is designed to help users assess their risk of a heart attack by inputting their health data.This is a Streamlit-based web application that predicts the likelihood of a heart attack based on patient data. The app uses a K-Nearest Neighbors (KNN) machine learning model trained on a dataset of heart disease patients.</p>


# Datasets Descriptions

The dataset used in this project is the **Heart Disease Dataset** from the UCI Machine Learning Repository. It contains the following features:

1. **Age**: Age of the patient.
2. **Sex**: Gender of the patient (1 = Male, 0 = Female).
3. **ChestPainType**: Type of chest pain (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic).
4. **RestingBP**: Resting blood pressure (mm Hg).
5. **Cholesterol**: Serum cholesterol level (mg/dl).
6. **FastingBS**: Fasting blood sugar (> 120 mg/dl: 1, < 120 mg/dl: 0).
7. **RestingECG**: Resting electrocardiographic results (Normal, ST-T Wave Abnormality, Left Ventricular Hypertrophy).
8. **MaxHR**: Maximum heart rate achieved.
9. **ExerciseAngina**: Exercise-induced angina (Yes: 1, No: 0).
10.**Oldpeak**: ST depression induced by exercise.
11.**ST_Slope**: Slope of the peak exercise ST segment (Upsloping, Flat, Downsloping).
12.**HeartDisease**: Target variable (1 = Heart Disease, 0 = No Heart Disease).

# Overview

The goal of this project is to predict the likelihood of a heart attack based on patient data. We explore and preprocess the dataset, train multiple machine learning models, evaluate their performance, and deploy the best model.


# Methodology
This project contains two .py files. The training and deploy files are Heart_Attack_Predictions.py and Heart_Attack_App_deploy.py respectively. The flow of the projects are as follows:

   ## 1. Importing the libraries and dataset

   The data are loaded from the dataset and usefull libraries are imported.

   ## 2. Exploratory data analysis

   We perform EDA to understand the dataset:

   - Check for missing values.

   - Visualize the distribution of features.

   - Analyze correlations between features.

   - Explore the target variable (HeartDisease).

   ## 3. Data Preprocessing
   
   1. Handling Categorical Variables:
      One-hot encode for the categorical variables.
   2. Scaling Numerical Features:
      Use StandardScaler to scale the numerical features.
   3. Train-Test Split:
      Split the dataset into training (80%) and testing (20%) sets.


   ## 4. Machine learning model 

   We train and evaluate the following models:

   1. Logistic Regression
   2. Random Forest
   3. Decision Tree
   4. K-Nearest Neighbors (KNN)
   5. Support Vector Classifier (SVC)
   6. Naive Bayes

   ## 5. Model Prediction and Accuracy

   Best Model: K-Nearest Neighbors (KNN) with an accuracy of 0.85 and ROC AUC of 0.90.
   
   - The Accuracy of models in a image.

     ![heart_accuracy](https://github.com/user-attachments/assets/3850c40e-9ddb-4f44-81d4-de04d075188d)


   ## 6. Model Deployment

  Save the trained model using `joblib` or `pickle`. Create a Streamlit app for user interaction.

   ## 7. Build the app using Streamlit

   The best model (KNN) is deployed as a web-based app using Streamlit. Users can input patient data and get predictions in real-time.
   (![heart_pred](https://github.com/user-attachments/assets/42fa2499-9de7-4956-a20f-f35c7983c326)
   
# Instructions for Running the App Locally

1. Clone this repository to your local machine.
2. Install the required libraries using pip install -r requirements.txt.
3. Run the Streamlit app using the command streamlit run app.py.
  
# Requirements

The main frameworks used in this project are Pandas, Matplotlib, Seaborn, Scikit-learn and Streamlit.
