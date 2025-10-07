# Disease Diagnosis Application

A machine learning–based application that takes user-entered symptoms and predicts a probable disease, along with guidance. This project demonstrates how to build a symptom-to-disease predictor using ML and a simple UI (web or desktop) interface.

## Overview
This application allows users to input their symptoms (from a predefined list) and uses a trained machine learning model to predict which disease is most likely. It also gives some guidance or suggestions based on the prediction. It can be used as a healthcare assistant, an educational ML prototype, or as part of a telemedicine or health chatbot.

## Features
- Accepts multiple symptoms as input  
- Uses a trained machine learning model (Decision Tree, Random Forest, or others)  
- Outputs the predicted disease  
- Provides guidance or basic advice  
- Simple UI for users (web or GUI)  
- Can be retrained or updated with new datasets  

## Architecture & Workflow
1. **Front-end Input** – User selects or enters symptoms.  
2. **Backend Processing** – Symptoms are encoded into numeric form and passed to the ML model.  
3. **Model Prediction** – Model predicts the most probable disease.  
4. **Response** – The predicted disease and advice are displayed.  

Workflow:  
User Input → Preprocessing → Model Inference → Prediction → Display Result

## Technologies & Dependencies
- **Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, joblib/pickle  
- **Framework (optional):** Flask or Django (for web), Tkinter/PyQt (for GUI)  
- **Version Control:** Git  
- **Dependencies File:** `requirements.txt`

**Description of Directories**  
- `model/`: stores trained model files  
- `data/`: stores datasets  
- `app.py`: main server/application file  
- `templates/` and `static/`: front-end files if using Flask/Django  

## Installation & Setup
### Prerequisites
- Python 3.7+ installed  
- pip package manager available  

### Steps
1. Clone the repository:  
   ```bash
   git clone https://github.com/gaonkarkavitaa/Disease-diagnosis-application.git
   cd Disease-diagnosis-application

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate        # macOS/Linux  
venv\Scripts\activate           # Windows

Install dependencies:

pip install -r requirements.txt

Usage

Launch the app and enter or select symptoms.

Click on Predict Disease.

The model predicts the most likely disease and displays it with optional guidance.

You can retrain the model by updating datasets and re-running the training script (if available).

Model Training & Dataset
Dataset

The dataset consists of symptom–disease pairs. Each row represents a patient’s symptoms (as binary or one-hot encoded values) and the disease label. Example file: data/symptoms_dataset.csv.

Model Training Process

Load the dataset using pandas.

Preprocess the data (handle nulls, encode features).

Split into training and testing sets.

Train the model using classifiers such as DecisionTree, RandomForest, or Naive Bayes.

Evaluate accuracy and confusion matrix.

Save the trained model as a .pkl file using joblib or pickle.

Example:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv('data/symptoms_dataset.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'model/trained_model.pkl')

Limitations & Assumptions

Works only with predefined symptoms in the dataset.

Not a replacement for professional medical diagnosis.

Accuracy depends on dataset quality and diversity.

No real-time updates or medical validation.

Model does not handle continuous or sequential symptoms.

Future Enhancements

Add more diseases and symptom variations.

Integrate patient history and demographics.

Include disease probability/confidence scores.

Add chatbot or voice-based symptom input.

Deploy to cloud (AWS, Azure, GCP) or mobile app.

Add an admin dashboard to manage datasets and retraining.
