import streamlit as st
import pandas as pd
import joblib

# Load saved model files
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Page setup
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

st.subheader("Customer Information")

# Numerical inputs
senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

# Customer details
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Prediction
if st.button("🔮 Predict Churn"):

    input_data = pd.DataFrame({
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],

        "gender_Male": [1 if gender == "Male" else 0],
        "Partner_Yes": [1 if partner == "Yes" else 0],
        "Dependents_Yes": [1 if dependents == "Yes" else 0],
        "PhoneService_Yes": [1 if phone_service == "Yes" else 0],

        "MultipleLines_No phone service":
            [1 if multiple_lines == "No phone service" else 0],
        "MultipleLines_Yes":
            [1 if multiple_lines == "Yes" else 0],

        "InternetService_Fiber optic":
            [1 if internet_service == "Fiber optic" else 0],
        "InternetService_No":
            [1 if internet_service == "No" else 0],

        "OnlineSecurity_No internet service":
            [1 if online_security == "No internet service" else 0],
        "OnlineSecurity_Yes":
            [1 if online_security == "Yes" else 0],

        "OnlineBackup_No internet service":
            [1 if online_backup == "No internet service" else 0],
        "OnlineBackup_Yes":
            [1 if online_backup == "Yes" else 0],

        "DeviceProtection_No internet service":
            [1 if device_protection == "No internet service" else 0],
        "DeviceProtection_Yes":
            [1 if device_protection == "Yes" else 0],

        "TechSupport_No internet service":
            [1 if tech_support == "No internet service" else 0],
        "TechSupport_Yes":
            [1 if tech_support == "Yes" else 0],

        "StreamingTV_No internet service":
            [1 if streaming_tv == "No internet service" else 0],
        "StreamingTV_Yes":
            [1 if streaming_tv == "Yes" else 0],

        "StreamingMovies_No internet service":
            [1 if streaming_movies == "No internet service" else 0],
        "StreamingMovies_Yes":
            [1 if streaming_movies == "Yes" else 0],

        "Contract_One year":
            [1 if contract == "One year" else 0],
        "Contract_Two year":
            [1 if contract == "Two year" else 0],

        "PaperlessBilling_Yes":
            [1 if paperless_billing == "Yes" else 0],

        "PaymentMethod_Credit card (automatic)":
            [1 if payment_method == "Credit card (automatic)" else 0],
        "PaymentMethod_Electronic check":
            [1 if payment_method == "Electronic check" else 0],
        "PaymentMethod_Mailed check":
            [1 if payment_method == "Mailed check" else 0]
    })

    # Match training feature order
    input_data = input_data[feature_names]

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")
    else:
        st.success("✅ Customer is likely to STAY.")