import streamlit as st
import joblib

st.title("🌧️ Rainfall Prediction")

model = joblib.load("decision_tree_model.pkl")

avg_temp = st.number_input("Average Temperature")
min_temp = st.number_input("Minimum Temperature")
max_temp = st.number_input("Maximum Temperature")
wind_speed = st.number_input("Wind Speed")
air_pressure = st.number_input("Air Pressure")
elevation = st.number_input("Elevation")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

if st.button("Predict"):
    x = [[avg_temp, min_temp, max_temp, wind_speed,
          air_pressure, elevation, latitude, longitude]]

    prediction = model.predict(x)[0]

    if prediction == 1:
        st.success("🌧️ Rain Expected")
    else:
        st.info("☀️ No Rain Expected")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Rainfall Prediction - Graphs")

# Load dataset
df = pd.read_excel("india_weather_rainfall_data.xlsx")

# Show dataset
st.dataframe(df.head())

# 1. Rainfall by Month
st.subheader("Rainfall by Month")
m = df.groupby("month")["rainfall"].mean()
st.bar_chart(m)

# 2. Rainfall by Season
st.subheader("Rainfall by Season")
s = df.groupby("season")["rainfall"].mean()
st.bar_chart(s)

# 3. Rainfall by State
st.subheader("Rainfall by State")
state = df.groupby("state")["rainfall"].mean().sort_values(ascending=False).head(10)
st.bar_chart(state)

# 4. Temperature vs Rainfall
st.subheader("Temperature vs Rainfall")
fig, ax = plt.subplots()
ax.scatter(df["avg_temp"], df["rainfall"], alpha=0.3)
ax.set_xlabel("Average Temperature")
ax.set_ylabel("Rainfall")
st.pyplot(fig)

# 5. Rainfall Distribution
st.subheader("Rainfall Distribution")
fig, ax = plt.subplots()
ax.hist(df["rainfall"].dropna(), bins=30)
ax.set_xlabel("Rainfall")
ax.set_ylabel("Frequency")
st.pyplot(fig)


