# # Sales Prediction Using Linear Regression
#
# # Importing the dependencies
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
#
# # Loading Dataset
# dataset = pd.read_csv("Advertising.csv")
# # print(dataset.head())
#
# dataset = dataset.drop(columns='Unnamed: 0',axis=1)
# # print(dataset.shape)
# # print(dataset.describe())
#
# # Visualizing the data
# plt.figure(figsize=(8,8))
# plt.scatter(dataset.TV,dataset.Sales)
# plt.title('TV Advertisement VS Sales')
# plt.xlabel('TV Advertisement')
# plt.ylabel('Sales')
# plt.show()
#
# plt.figure(figsize=(8,8))
# plt.scatter(dataset.Radio,dataset.Sales)
# plt.title('TV Advertisement VS Sales')
# plt.xlabel('TV Advertisement')
# plt.ylabel('Sales')
# plt.show()
#
# plt.figure(figsize=(8,8))
# plt.scatter(dataset.Newspaper,dataset.Sales)
# plt.title('TV Advertisement VS Sales')
# plt.xlabel('TV Advertisement')
# plt.ylabel('Sales')
# plt.show()
#
# # Visualizing the correlation between feature of data through heatmap
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize=(12,10))
# sns.heatmap(dataset.corr())
#
# # Splitting the dataset into features and target
# feature = dataset.drop(columns='Sales',axis=1)
# target = dataset['Sales']
# # print(feature)
# # print(target)
#
# # Splitting data into training and testing data
# x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size=0.3,random_state=4)
#
# # Selecting Model and Training it
# model = LinearRegression()
# model.fit(x_train,y_train)
#
# y_pred = model.predict(x_test)
#
# print(f'Mean Squared Error : {(metrics.mean_absolute_error(y_pred,y_test)).round(5)}')
# print(f'R-Squared error : {(metrics.r2_score(y_pred,y_test)*100).round(2)} %')
#
# print(model.predict([[230.1,37.8,69.2]]))


# Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import streamlit as st

# Loading Dataset
dataset = pd.read_csv("Advertising.csv")
dataset = dataset.drop(columns='Unnamed: 0', axis=1)

# Define the Streamlit app
def main():
    st.sidebar.title("Sales Prediction")
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an option", ("Exploratory Data Analysis", "Train Model", "Make Prediction"))

    if option == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        visualize_data(dataset)
    elif option == "Train Model":
        st.header("Train Model")
        train_model(dataset)
    elif option == "Make Prediction":
        st.header("Make Prediction")
        make_prediction()

# Function for visualizing the data
def visualize_data(dataset):
    st.subheader("Visualizing the data")
    plt.figure(figsize=(8, 8))
    plt.scatter(dataset.TV, dataset.Sales)
    plt.title('TV Advertisement VS Sales')
    plt.xlabel('TV Advertisement')
    plt.ylabel('Sales')
    st.pyplot()

    plt.figure(figsize=(8, 8))
    plt.scatter(dataset.Radio, dataset.Sales)
    plt.title('Radio Advertisement VS Sales')
    plt.xlabel('Radio Advertisement')
    plt.ylabel('Sales')
    st.pyplot()

    plt.figure(figsize=(8, 8))
    plt.scatter(dataset.Newspaper, dataset.Sales)
    plt.title('Newspaper Advertisement VS Sales')
    plt.xlabel('Newspaper Advertisement')
    plt.ylabel('Sales')
    st.pyplot()

    st.subheader("Correlation between features")
    plt.figure(figsize=(12, 10))
    sns.heatmap(dataset.corr(), annot=True)
    st.pyplot()

# Function for training the model
def train_model(dataset):
    # Splitting the dataset into features and target
    feature = dataset.drop(columns='Sales', axis=1)
    target = dataset['Sales']

    # Splitting data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=4)

    # Selecting Model and Training it
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(x_test)

    st.subheader("Model Performance")
    st.write(f'Mean Squared Error: {(metrics.mean_absolute_error(y_pred, y_test)).round(5)}')
    st.write(f'R-Squared Error: {(metrics.r2_score(y_pred, y_test) * 100).round(2)} %')

# Function for making predictions
def make_prediction():
    st.subheader("Enter the advertising budget")
    tv = st.number_input("TV Advertisement", min_value=0.0, step=0.1, format="%.1f")
    radio = st.number_input("Radio Advertisement", min_value=0.0, step=0.1, format="%.1f")
    newspaper = st.number_input("Newspaper Advertisement", min_value=0.0, step=0.1, format="%.1f")

    # Loading Dataset
    dataset = pd.read_csv("Advertising.csv")
    dataset = dataset.drop(columns='Unnamed: 0', axis=1)

    # Splitting the dataset into features and target
    feature = dataset.drop(columns='Sales', axis=1)
    target = dataset['Sales']

    # Training the model on the entire dataset
    model = LinearRegression()
    model.fit(feature, target)

    # Making the prediction
    prediction = model.predict([[tv, radio, newspaper]])

    st.subheader("Sales Prediction")
    if st.button("Predict"):
        st.write(f"The predicted sales for the given advertising budget are: {prediction[0]:.2f}")

# Run the app
if __name__ == "__main__":
    main()
