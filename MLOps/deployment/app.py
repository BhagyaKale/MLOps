import streamlit as st
import pandas as pd
import joblib
import os

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package App")
st.write("Kindly enter the details to check whether they are likely to purchase the package.")

# Collect user input
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Free Lancer", "Small Business", "Large Business", "Unemployed", "Student"])
TypeofContact = st.selectbox("Type of Contact (How was the customer contacted)", ["Self Enquiry", "Company Invited", "No Contact"]) # Added "No Contact" based on data in data_register.py
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier (customer's city tier)", ["Tier 1", "Tier 2", "Tier 3"])
DurationofPitch = st.number_input("Duration of Pitch (duration of the pitch in minutes)", min_value=1, value=15)
PreferredPropertyStar = st.number_input("Preferred Property Star (customer's preferred property star rating)", min_value=1, max_value=5, value=4)
NumberOfFollowups = st.number_input("Number of Follow-ups (number of follow-ups the customer has had)", min_value=0, value=2)
Gender = st.selectbox("Gender (customer's gender)", ["Male", "Female", "Other"]) # Added "Other" for completeness
NumberOfTrips = st.number_input("NumberOfTrips (number of trips the customer has made)", min_value=1, value=2)
Passport = st.selectbox("Passport (customer's passport ownership)", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (customer's pitch satisfaction score)", min_value=1, max_value=5, value=4)
OwnCar = st.selectbox("Own Car (customer's ownership of a car)", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (number of children the customer is visiting with)", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income (customer's monthly income)", min_value=0, value=5000)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (number of persons the customer is visiting)", min_value=1, value=2)
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Married", "Single", "Divorced", "Unmarried"]) # Added "Divorced", "Unmarried"
Designation = st.selectbox("Designation (customer's designation)", ["Executive", "Managerial", "Professional", "Other", "AVP", "Senior Manager"])
ProductPitched = st.selectbox("Product Pitched (product pitched by the customer)", ["Business Package", "Deluxe Package", "Standard Package", "Super Deluxe Package", "King Package"])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    "Occupation": Occupation,
    "TypeofContact": TypeofContact,
    "Age": float(Age),
    "CityTier": float(CityTier.split(' ')[1]),
    "DurationOfPitch": float(DurationofPitch),
    "PreferredPropertyStar": float(PreferredPropertyStar),
    "NumberOfFollowups": float(NumberOfFollowups),
    "Gender": Gender,
    "NumberOfTrips": float(NumberOfTrips),
    "Passport": 1.0 if Passport == "Yes" else 0.0,
    "PitchSatisfactionScore": float(PitchSatisfactionScore),
    "OwnCar": 1.0 if OwnCar == "Yes" else 0.0,
    "NumberOfChildrenVisiting": float(NumberOfChildrenVisiting),
    "MonthlyIncome": float(MonthlyIncome),
    "NumberOfPersonVisiting": float(NumberOfPersonVisiting),
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the package" if prediction == 1 else "not purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
