import streamlit as st
import pandas as pd
import joblib
import numpy as np
st.set_page_config(
    page_title="HousePrice AI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 HousePrice AI")
st.subheader("California Housing Price Prediction")

st.write(
    "An End-to-End Machine Learning Project "
    "to Predict Median House Values."
)
st.info(
    "Welcome to HousePrice AI! This application uses "
    "Machine Learning to predict California median "
    "house values using housing and location features."
)
st.divider()

st.subheader("🛠️ Technologies Used")

st.write(
    "Python | Pandas | NumPy | Scikit-learn | "
    "Matplotlib | Seaborn | Streamlit"
)
st.divider()
model = joblib.load("models/random_forest_model.pkl")

preprocessor = joblib.load("models/preprocessor.pkl")
st.subheader("🚀 Project Highlights")

col1, col2, col3 = st.columns(3)

col1.info("📊 20,640 Housing Records")
col2.info("🤖 7 Model Comparisons")
col3.info("🎯 81.22% R² Score")
st.sidebar.header("Enter Housing Details")

longitude = st.sidebar.number_input(
    "Longitude", value=-122.23
)

latitude = st.sidebar.number_input(
    "Latitude", value=37.88
)

housing_median_age = st.sidebar.number_input(
    "Housing Median Age", min_value=1, value=30
)

median_income = st.sidebar.number_input(
    "Median Income (in tens of thousands)",
    min_value=0.0,
    value=3.5
)

total_rooms = st.sidebar.number_input(
    "Total Rooms", min_value=1, value=2000
)
total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms", min_value=1, value=400
)

population = st.sidebar.number_input(
    "Population", min_value=1, value=1000
)

households = st.sidebar.number_input(
    "Households", min_value=1, value=350
)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    input_data["rooms_per_household"] = (
        input_data["total_rooms"] / input_data["households"]
    )

    input_data["bedrooms_per_room"] = (
        input_data["total_bedrooms"] / input_data["total_rooms"]
    )

    input_data["population_per_household"] = (
        input_data["population"] / input_data["households"]
    )
    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)

    st.success("Prediction Generated Successfully!")

    st.metric(
        label="Predicted Median House Value",
        value=f"${prediction[0]:,.2f}"
    )
    st.divider()

st.caption(
    "Disclaimer: This model predicts median house values "
    "for California census-block areas. It is not an "
    "individual property valuation tool."
)
st.divider()

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", "81.22%")
col2.metric("MAE", "$31,889.92")
col3.metric("RMSE", "$49,610.01")

st.caption("Best Model: Tuned Random Forest Regressor")
st.divider()

st.header("📈 Model Comparison")

model_results = pd.DataFrame({
    "Model": [
        "Multiple Linear Regression",
        "Simple Linear Regression",
        "Polynomial Regression",
        "SVR",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "R2 Score": [
        0.6353,
        0.4589,
        0.4633,
        0.4431,
        0.7065,
        0.8062,
        0.8122
    ]
})

st.bar_chart(
    model_results.set_index("Model")
)
st.divider()

st.header("⚠️ Project Limitations")

st.write("""
- The model uses the California Housing dataset.
- Predictions represent median values of census-block areas,
  not individual houses.
- The dataset target values are capped at $500,001.
- Actual market prices may differ from predictions.
""")