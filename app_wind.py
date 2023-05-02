import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


def print_result(entries):
    # Get user data from the entries
    user_data = get_user_data(entries)
    # Create a dictionary with the user data and attribute names
    attributes = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level',
                  'blood_glucose_level']
    data_dict = dict(zip(attributes, user_data))

    # Load label encoders for gender and smoking history
    label_encoder_gender = joblib.load('encoder_gen.joblib')
    label_encoder_smoking_history = joblib.load('encoder_smok.joblib')
    # Transform gender and smoking history using the label encoders
    data_dict['gender'] = label_encoder_gender.transform([data_dict['gender']])
    data_dict['smoking_history'] = label_encoder_smoking_history.transform([data_dict['smoking_history']])

    # Convert dictionary values to appropriate types
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.size == 1:
            data_dict[key] = np.squeeze(value).item()
        else:
            data_dict[key] = value
    # Convert dictionary to a 2D numpy array
    input_array = np.array(list(data_dict.values()))
    input_array = input_array.reshape(1, -1)

    # Load the diabetes prediction model and predict the probability of diabetes
    diabetes_model = joblib.load('dia_pred_mod.pkl')
    result = diabetes_model.predict_proba(input_array)[0][1]
    formatted_result = "{:.6f}".format(result)
    # Clear the result entry and insert the formatted result
    result_entry.delete(0, tk.END)
    result_entry.insert(0, formatted_result)


def close_info_window(info_window):
    # Destroy the information window
    info_window.destroy()


def info_func():
    # Create a new window for information
    info_window = tk.Toplevel(root)
    info_window.title("Info")
    # Add a label with information text to the new window
    information = "Diabetes Prediction Program\n\n" \
                  "This program allows you to predict the probability of having diabetes based on some personal information. To use the program, please follow these steps:\n\n" \
                  "- Enter your personal information in the fields provided. The fields are as follows:\n" \
                  "  Gender: Enter your gender (Male or Female)\n" \
                  "  Age: Enter your age in years\n" \
                  "  Hypertension: Enter 1 if you have hypertension, or 0 otherwise\n" \
                  "  Heart disease: Enter 1 if you have heart disease, or 0 otherwise\n" \
                  "  Smoking history: Enter 1 if you have a history of smoking, or 0 otherwise\n" \
                  "  BMI: Enter your body mass index (BMI), which is a measure of body fat based on height and weight\n" \
                  "  HbA1c level: Enter your HbA1c level, which is a measure of average blood sugar levels over the past 2-3 months\n" \
                  "  Blood glucose level: Enter your current blood glucose level in mg/dL\n\n" \
                  "- Please note that all fields are required.\n\n" \
                  "Once you have entered your personal information, click on the \"Result\" button. The program will calculate the probability of having diabetes based on your information.\n\n" \
                  "The result will be displayed in the \"Result\" field, which shows the percentage of the diabetes probability.\n\n" \
                  "- Please note that the diabetes prediction is based on a statistical model and is not a substitute for medical advice. If you are concerned about your health, please consult a healthcare professional."
    tk.Label(info_window, text=information, justify='left', wraplength=400).pack(padx=10, pady=10)
    # Add a close button to the new window
    close_button = tk.Button(info_window, text="Close", command=lambda: close_info_window(info_window))
    close_button.pack(padx=10, pady=10)


def get_user_data(entries):
    user_data = []
    # Get info from each line
    for entry in entries:
        value = entry.get()
        if value.isdigit():
            user_data.append(float(value))
        else:
            user_data.append(value)
    # Return an array
    return user_data


# Creating main window
root = tk.Tk()
root.title("Diabetes Prediction")

# Creating lists of variants
data = pd.read_csv("res/diabetes_prediction_dataset.csv")
gender_values = data['gender'].unique().tolist()
hypertension_values = data['hypertension'].unique().tolist()
heart_disease_values = data['heart_disease'].unique().tolist()
smoking_history_values = data['smoking_history'].unique().tolist()

# create 9 label and entry widgets
labels = ['Gender', 'Age', 'Hypertension', 'Heart disease', 'Smoking history', 'Bmi', 'HbA1c level',
          'Blood Glucose Level', 'Result']
entries = []
for i in range(8):
    tk.Label(root, text=labels[i]).grid(row=i, column=0, padx=10, pady=10)
    if labels[i] == 'Gender':
        gender_var = tk.StringVar()
        gender_var.set(gender_values[0])
        entry = ttk.Combobox(root, textvariable=gender_var, values=gender_values)
    elif labels[i] == 'Hypertension':
        hypertension_var = tk.StringVar()
        hypertension_var.set(hypertension_values[0])
        entry = ttk.Combobox(root, textvariable=hypertension_var, values=hypertension_values)
    elif labels[i] == 'Heart disease':
        heart_disease_var = tk.StringVar()
        heart_disease_var.set(heart_disease_values[0])
        entry = ttk.Combobox(root, textvariable=heart_disease_var, values=heart_disease_values)
    elif labels[i] == 'Smoking history':
        smoking_history_var = tk.StringVar()
        smoking_history_var.set(smoking_history_values[0])
        entry = ttk.Combobox(root, textvariable=smoking_history_var, values=smoking_history_values)
    else:
        entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=10)
    entries.append(entry)

# Result field
result_label = tk.Label(root, text=labels[8])
result_label.grid(row=8, column=0, padx=10, pady=10)
result_entry = tk.Entry(root)
result_entry.grid(row=8, column=1, padx=10, pady=10)


# Create a button to print the result
result_button = tk.Button(root, text="Print result", command=lambda: print_result(entries))
result_button.grid(row=9, column=1, columnspan=1, padx=5, pady=5)

# Create a button to print the info
info_button = tk.Button(root, text="Info", command=info_func)
info_button.grid(row=9, column=0, columnspan=1, padx=5, pady=5)

root.mainloop()