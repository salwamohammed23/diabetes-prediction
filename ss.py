
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')
# printing the first 5 rows of the dataset
diabetes_dataset.head()

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
# training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Saving the trained model
#filename = 'trained_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Giving title
    st.title('Diabetes Prediction App')
    # Getting the input data
    Pregnancies = st.text_input('No. of Pregnancies')
    Glucose = st.text_input('No. of Glucose')
    BloodPressure = st.text_input('No. of Blood Pressure')
    SkinThickness = st.text_input('No. of Skin Thickness')
    Insulin = st.text_input('No. of Insulin')
    BMI = st.text_input('No. of BMI')
    DiabetesPedigreeFunction = st.text_input('No. of Diabetes Pedigree Function')
    Age = st.text_input('No. of Age')

    # Code for prediction
    diagnosis = ""
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()