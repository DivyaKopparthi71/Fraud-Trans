import streamlit as st
import pickle
import pandas as pd

# Load the model
try:
    model_xgb = pickle.load(open('fraud_final.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Set up the title for the Streamlit app
st.title("Fraud Detection Model")

# Function to collect user inputs
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
    
    features = pd.DataFrame([data])
    features['type'] = features['type'].map({'CASH_OUT': 0, 'TRANSFER': 1})  # Update mapping as needed
    
    return features

# Get user inputs
input_df = user_input_features()

# Predict and display results
if st.button("Predict Fraud"):
    try:
        prediction = model_xgb.predict(input_df)
        prediction_proba = model_xgb.predict_proba(input_df)[:, 1]

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.write("This transaction is predicted to be **Fraudulent**.")
        else:
            st.write("This transaction is predicted to be **Not Fraudulent**.")
        
        st.subheader("Prediction Probability")
        st.write(f"Probability of Fraud: {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
