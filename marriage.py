import streamlit as st
import pandas as pd
import joblib  # For loading your trained model

# Load the trained marriage proposal model
model = joblib.load('marriage_proposal_model.pkl')

# Load dataset to validate possible input values
df = pd.read_csv('marriage_proposal.csv')

# Extract potential min/max ranges from the dataset (for validation)
min_height, max_height = df['Height'].min(), df['Height'].max()
min_age, max_age = df['Age'].min(), df['Age'].max()
min_income, max_income = df['Income'].min(), df['Income'].max()
min_romantic_gesture_score, max_romantic_gesture_score = df['RomanticGestureScore'].min(), df['RomanticGestureScore'].max()
min_compatibility_score, max_compatibility_score = df['CompatibilityScore'].min(), df['CompatibilityScore'].max()
min_communication_score, max_communication_score = df['CommunicationScore'].min(), df['CommunicationScore'].max()
min_distance_km, max_distance_km = df['DistanceKM'].min(), df['DistanceKM'].max()

# Function to take user input from UI
def user_input_features():
    height = st.number_input("Height (in cm)", min_value=int(min_height), max_value=int(max_height), value=int((min_height+max_height)//2))
    age = st.number_input("Age", min_value=int(min_age), max_value=int(max_age), value=int((min_age+max_age)//2))
    income = st.number_input("Income (in $)", min_value=int(min_income), max_value=int(max_income), value=int((min_income+max_income)//2))
    romantic_gesture_score = st.number_input("Romantic Gesture Score", min_value=int(min_romantic_gesture_score), max_value=int(max_romantic_gesture_score), value=int((min_romantic_gesture_score+max_romantic_gesture_score)//2))
    compatibility_score = st.number_input("Compatibility Score", min_value=int(min_compatibility_score), max_value=int(max_compatibility_score), value=int((min_compatibility_score+max_compatibility_score)//2))
    communication_score = st.number_input("Communication Score", min_value=int(min_communication_score), max_value=int(max_communication_score), value=int((min_communication_score+max_communication_score)//2))
    distance_km = st.number_input("Distance (in KM)", min_value=int(min_distance_km), max_value=int(max_distance_km), value=int((min_distance_km+max_distance_km)//2))

    # Dropdown for Age Category
    age_category = st.selectbox(
        "Age Category",
        options=["Middle-aged", "Senior", "Young", "Not available"]
    )

    # Mapping AgeCategory to binary features (one-hot encoding)
    age_category_middle_aged = 1 if age_category == "Middle-aged" else 0
    age_category_senior = 1 if age_category == "Senior" else 0
    age_category_young = 1 if age_category == "Young" else 0
    age_category_not_available = 1 if age_category == "Not available" else 0

    # Dictionary to hold user inputs
    data = {
        'Height': height,
        'Age': age,
        'Income': income,
        'RomanticGestureScore': romantic_gesture_score,
        'CompatibilityScore': compatibility_score,
        'CommunicationScore': communication_score,
        'DistanceKM': distance_km,
        'AgeCategory_Middle-aged': age_category_middle_aged,
        'AgeCategory_Senior': age_category_senior,
        'AgeCategory_Young': age_category_young,
        'AgeCategory_not available': age_category_not_available  # Fix: Match the exact feature name used during training
    }
    
    # Convert the data into a DataFrame for model prediction
    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app title
st.title('Marriage Proposal Prediction')

# Get user inputs
input_df = user_input_features()

# Perform prediction when the button is clicked
if st.button('Predict Proposal Outcome'):
    prediction = model.predict(input_df)

    # Check the response: 1 for accepted, 0 for rejected
    if prediction == 1:
        st.success('Proposal Accepted!')
    else:
        st.error('Proposal Rejected.')
