import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="vishalverma24/tourism-package-prediction-ui", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the sensor and configuration data below to get a prediction.
""")


# User input
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Large Business", "Small Business"])
Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "King", "Super Deluxe"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Unmarried", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

Age = st.number_input("Age", min_value=0, max_value=100, value=50, step=1)
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0, max_value=1000, value=5, step=1)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=10, value=3, step=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=3, step=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=4, step=1)
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=50, value=5, step=1)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.number_input("Own Car", min_value=0, max_value=1, value=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=2)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=10000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])

if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
