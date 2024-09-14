import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load the model
model_xgb = pickle.load(open('fraud_final.pkl', 'rb'))

# Set up the title for the Streamlit app
st.title("Fraud Detection Model")

# Create a function to take user inputs and make predictions
def user_input_features():
    step = st.number_input("Step (Time in hours)", min_value=1, step=1)
    txn_type = st.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER'])  # Adjust based on your dataset
    amount = st.number_input("Transaction Amount", min_value=0.0)
    nameOrig = st.text_input("Original Account Holder")
    oldbalanceOrg = st.number_input("Original Account Balance", min_value=0.0)
    newbalanceOrig = st.number_input("New Account Balance", min_value=0.0)
    
    data = {
        'step': step,
        'type': txn_type,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig
    }
    
    # Convert the user input into a dataframe for the model
    features = pd.DataFrame([data])
    
    # Handle any encoding if needed (e.g., converting type into numerical value)
    features['type'] = features['type'].map({'CASH_OUT': 0, 'TRANSFER': 1})  # Update mapping as per your dataset
    
    return features

# Get the user input features
input_df = user_input_features()

# When the "Predict" button is clicked, make a prediction
if st.button("Predict Fraud"):
    # Make predictions with the input data
    prediction = model_xgb.predict(input_df)
    prediction_proba = model_xgb.predict_proba(input_df)[:, 1]

    # Display the prediction and probability
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.write("This transaction is predicted to be **Fraudulent**.")
    else:
        st.write("This transaction is predicted to be **Not Fraudulent**.")
    
    st.subheader("Prediction Probability")
    st.write(f"Probability of Fraud: {prediction_proba[0]:.2f}")
