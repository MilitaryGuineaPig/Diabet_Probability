import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the data
data = pd.read_csv("res/diabetes_prediction_dataset.csv")

print(data.columns)

# Drop rows with missing values
data.dropna(inplace=True)

# Initialize the LabelEncoder object
label_encoder_gender = LabelEncoder()
label_encoder_smoking_history = LabelEncoder()
data['gender'] = label_encoder_gender.fit_transform(data['gender'])
data['smoking_history'] = label_encoder_smoking_history.fit_transform(data['smoking_history'])
joblib.dump(label_encoder_gender, 'encoder_gen.joblib')
joblib.dump(label_encoder_smoking_history, 'encoder_smok.joblib')

# Split the dataset into features and target variable
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
logreg = LogisticRegression(random_state=1, max_iter=10000)
logreg.fit(X_train, y_train)

# Evaluate the trained model on the testing data
accuracy = logreg.score(X_test, y_test)
print('Test accuracy:', accuracy)

# Save the trained model as a pickle file
joblib.dump(logreg, 'dia_pred_mod.pkl')