
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

# Split the dataset into features and target variable
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train)

# Train an SVM model on the training data
svm = SVC(kernel='linear', C=1, random_state=1)
svm.fit(X_train, y_train)

# Evaluate the trained model on the testing data
accuracy = svm.score(X_test, y_test)
print('Test accuracy:', accuracy)

# Save the trained model as a pickle file
joblib.dump(svm, 'dia_pred_svm_mod.pkl')
