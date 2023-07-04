# Iris flower classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=4)

# Train the K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Create a web app using Streamlit
st.title("Iris Flower Classification")
st.write("Enter the values for four different features to classify the flower:")

# Create input fields for user to enter values
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Create a button to trigger classification
classify_button = st.button("Classify")

# Perform classification when the button is clicked
if classify_button:
    # Create a new data point from user input
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the class label using the trained model
    prediction = knn.predict(new_data)
    predicted_class = data.target_names[prediction[0]]

    # Display the predicted class label
    st.write(f"Predicted Class: {predicted_class}")
