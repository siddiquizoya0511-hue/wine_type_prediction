import numpy as np
import joblib
import streamlit as st

model = joblib.load('model.pkl')

st.set_page_config(page_title='Wine Type Prediction', layout="centered")
st.title('Wine Type Classification App')
st.write("Predict whether the wine is **Red** or **White** using chemical properties")

fixed_acidity = st.number_input("Fixed acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile acidity", min_value=0.0)
citric_acid = st.number_input("Citric acid", min_value=0.0)
residual_sugar = st.number_input("Residual sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free sulfur dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total sulfur dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)
quality = st.number_input("Quality", min_value=0)

if st.button('Predict'):
    input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        quality
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error('Red Wine 🍷')
    else:
        st.success('White Wine 🥂')
