import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data = pd.read_csv("res/diabetes_prediction_dataset.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Initialize the LabelEncoder object
label_encoder_gender = joblib.load('encoder_gen.joblib')
label_encoder_smoking_history = joblib.load('encoder_smok.joblib')
data['gender'] = label_encoder_gender.transform(data['gender'])
data['smoking_history'] = label_encoder_smoking_history.transform(data['smoking_history'])
data = data.astype(float)


