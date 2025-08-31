import joblib
import streamlit as st
import pandas as pd

# Load trained model
model = joblib.load("fraud_detection_model.jb")

# Load LabelEncoder (saved during training)
encoder = joblib.load("label_encoder.jb")

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")
st.markdown("Please enter the following transaction details:")

st.divider()

# User inputs
transaction_type = st.selectbox(
    "Transaction Type", 
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, max_value=100000000.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account (Sender)", min_value=0.0, step=0.1, max_value=10000000.0)
newbalanceOrig = st.number_input("New Balance of Origin Account (Sender)", min_value=0.0, step=0.1, max_value=10000000.0)
oldbalanceDest = st.number_input("Old Balance of Destination Account (Receiver)", min_value=0.0, step=0.1, max_value=10000000.0)
newbalanceDest = st.number_input("New Balance of Destination Account (Receiver)", min_value=0.0, step=0.1, max_value=10000000.0)

# Prediction button
if st.button("Predict Fraud"):
    # Create dataframe
    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "balanceDiff": [oldbalanceOrg - newbalanceOrig],
        "balancediffDES": [oldbalanceDest - newbalanceDest]
    })

    # Encode transaction type
    input_data["type"] = encoder.transform(input_data["type"])

    # Predict
    prediction = model.predict(input_data)

    st.subheader(f"Prediction Result: {int(prediction)}")
    
    if prediction[0] == 1:
        st.error("🚨 The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ The transaction is predicted to be LEGITIMATE.")

    
 