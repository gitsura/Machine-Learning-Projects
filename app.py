# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Delivery Time Predictor", layout="wide")
st.title("🚀 Delivery Time Predictor")

# Sample dataset (for training)
data = {
    "City": ["CityA", "CityB", "CityA", "CityC", "CityB"],
    "Type of Vehicle": ["Bike", "Car", "Bike", "Bike", "Car"],
    "Type of Order": ["Food", "Grocery", "Food", "Food", "Grocery"],
    "Distance": [5, 3, 7, 6, 4],  # in km
    "Weather": ["Sunny", "Rainy", "Sunny", "Cloudy", "Rainy"],
    "Time Taken": [30, 25, 40, 35, 28]  # in minutes
}
df = pd.DataFrame(data)

# Features and target
X = df.drop("Time Taken", axis=1)
y = df["Time Taken"]

# Selected features
categorical_features = ["City", "Type of Vehicle", "Type of Order", "Weather"]
numeric_features = ["Distance"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X, y)

# Sidebar: user input
st.sidebar.header("Enter Delivery Details")

city = st.sidebar.text_input("City", "")
vehicle = st.sidebar.text_input("Type of Vehicle", "")
order_type = st.sidebar.text_input("Type of Order", "")
weather = st.sidebar.text_input("Weather", "")
distance = st.sidebar.number_input("Distance (km)", min_value=0, value=5)

# Predict only if required fields are filled
if city and vehicle and order_type and weather:
    input_df = pd.DataFrame([{
        "City": city,
        "Type of Vehicle": vehicle,
        "Type of Order": order_type,
        "Weather": weather,
        "Distance": distance
    }])

    predicted_time = model.predict(input_df)[0]

    st.subheader("Predicted Delivery Time")
    st.write(f"⏱️ The estimated delivery time is **{round(predicted_time, 2)} minutes**")
else:
    st.warning("Please fill all fields to get a prediction.")

