import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# Load the dataset
df = pd.read_csv("Downloads/rainfall_prediction_dataset.csv")

# Assuming 'rainfall' column contains the labels for rainfall prediction (0 or 1)
X = df.drop(columns=['rainfall'])  # Features
Y = df['rainfall']  # Target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(
    C=0.03039195382313198,
    max_iter=100,
    penalty = 'l2',
    solver='liblinear',
    tol= 0.01
)

lr.fit(X_train_scaled, y_train)

# Streamlit App
logo_path = r"Downloads/rain_logo.png" 
st.image(logo_path, use_column_width='auto')

# Streamlit app title
st.title("Rainfall Prediction for Optimized Agricultural Operations")

# App description
st.write("""
Input the corresponding data below to predict if there will be rainfall.
""")

# Sidebar for user inputs
st.header('Input Parameters')

def user_input_features():
    pressure = st.number_input('Atmospheric Pressure ', min_value=0.0, max_value=1200.0, format="%.2f")
    maxtemp = st.number_input('Maximum Temperature ', min_value=0.0, max_value=40.0, format="%.2f")
    humidity = st.number_input('Humidity Level', min_value=0, max_value=100)  # Keep this as integer input
    dewpoint = st.number_input('Dew Point', min_value=-30.0, max_value=30.0, format="%.2f")
    cloud = st.number_input('Cloud level', min_value=0, max_value=100)  # Keep this as integer input
    sunshine = st.number_input('Sun intensity', min_value=0.0, max_value=50.0, format="%.2f")
    windspeed = st.number_input('Wind Speed', min_value=0.0, max_value=150.0, format="%.2f")
    
    data = {
        'pressure': pressure,
        'maxtemp': maxtemp,
        'humidity': humidity,
        'dewpoint': dewpoint,  
        'cloud': cloud,
        'sunshine': sunshine,
        'windspeed': windspeed
    }

    # Convert the input data into a DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Align the features with the training data (adding missing columns if necessary)
    features = features.reindex(columns=X_train.columns, fill_value=0)
    
    return features

input_df = user_input_features()

if st.button('Submit'):
    # Scale the user input
    input_scaled = scaler.transform(input_df)

    # Make predictions based on user input
    prediction = lr.predict(input_scaled)[0]  # Get 0 or 1 prediction
    prediction_proba = lr.predict_proba(input_scaled)  # Probability for each class
    
    prob_rain = prediction_proba[0][1]  # Probability of rain (positive class)
    prob_no_rain = prediction_proba[0][0]  # Probability of no rain (negative class)

    # Display the prediction results
    st.subheader('Prediction')
    if prediction == 1:
        st.write("Rain is expected.")
    else:
        st.write("No rain is expected.")
    
    st.subheader('Prediction Probability')
    st.write(f"Probability of Rain: {prob_rain:.2f}")
    st.write(f"Probability of No Rain: {prob_no_rain:.2f}")