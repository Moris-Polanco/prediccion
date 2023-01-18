import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("student_data.csv")

# Split the data into training and test sets
X = data.drop("success", axis=1)
y = data["success"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict student success
def predict_success(model, test_data):
    predictions = model.predict(test_data)
    return predictions

st.title("Predicting Student Success")

# Get user input
math_score = st.number_input("Math Score:", min_value=0, max_value=100)
reading_score = st.number_input("Reading Score:", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score:", min_value=0, max_value=100)

# Create a dataframe with the user input
user_data = {'math_score': [math_score], 'reading_score': [reading_score], 'writing_score': [writing_score]}
user_input = pd.DataFrame(user_data)

if st.button("Predict"):
    prediction = predict_success(model, user_input)
    if prediction == 1:
        st.success("The student is likely to be successful.")
    else:
        st.error("The student is unlikely to be successful.")
