import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle
import os

# Load the model
try:
    with open('optimized_best_model_LogisticRegression.pkl', 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()



# App layout and theme
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered",
    initial_sidebar_state="expanded"
)

# App title
st.title("Diabetes Prediction App")
st.markdown("Predict the chances of being diagnosed with diabetes based on medical inputs.")

# Inputs
st.sidebar.header("Patient Details")
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=100, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure Level", min_value=0, max_value=200, value=80, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=30, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30, step=1)

# Prediction button
if st.button("Predict"):
    try:
        # prepare input data for prediction
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure':[blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin' : [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age':[age]
        })




        # make prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.error(f"Patient Has diabetess. Probability: {probability:.2%}")
            else:
                st.success(f"Patient does not have diabetes. Probability: {probability:.2%}")
        else:
            prediction = model.predict(input_data)
            st.info(f"The model predicts a biabetes risk score of {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occoured during prediction: {e}")


    # # Prepare the input data for prediction
    
    
    # # Make the prediction
    # prediction = model.predict(input_data)
    # probability = model.predict_proba(input_data)[0][1]  # Probability of being diabetic

    # # Display the result
    # if prediction[0] == 1:
    #     st.error(f"The model predicts that the patient has diabetes. Probability: {probability:.2%}")
    # else:
    #     st.success(f"The model predicts that the patient does not have diabetes. Probability: {probability:.2%}")
