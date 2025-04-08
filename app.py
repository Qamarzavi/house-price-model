# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("House Price Prediction App")

st.sidebar.header("Enter House Features")

def user_input():
    data = {
        'OverallQual': st.sidebar.slider('Overall Quality (1-10)', 1, 10, 5),
        'GrLivArea': st.sidebar.slider('Living Area (sq ft)', 500, 4000, 1500),
        'GarageCars': st.sidebar.slider('Garage Cars', 0, 4, 1),
        'TotalBsmtSF': st.sidebar.slider('Basement Area (sq ft)', 0, 3000, 800),
        'YearBuilt': st.sidebar.slider('Year Built', 1900, 2022, 2000),
        '1stFlrSF': st.sidebar.slider('1st Floor SF', 500, 3000, 1000),
        'TotRmsAbvGrd': st.sidebar.slider('Total Rooms', 2, 14, 6)
    }
    return pd.DataFrame([data])

input_df = user_input()

# Predict
prediction = model.predict(input_df)

st.subheader("Predicted Sale Price")
st.write(f"${prediction[0]:,.2f}")
