import streamlit as st
import requests
import pandas as pd
import datetime

st.title("📈 Revenue Forecast App")

st.write("Enter features to predict revenue")

# Input form

event_date = st.date_input(
    "Event Date",
    value=datetime.date(2011, 9, 1),  # default value
    min_value=datetime.date(2009, 1, 1),  # your dataset start
    max_value=datetime.date(2012, 12, 31),  # optional upper bound
)

lag_1 = st.number_input("lag_1", value=30000.0)
lag_7 = st.number_input("lag_7", value=28000.0)
lag_14 = st.number_input("lag_14", value=26000.0)
lag_28 = st.number_input("lag_28", value=25000.0)

rolling_mean_7 = st.number_input("rolling_mean_7", value=29000.0)
rolling_std_7 = st.number_input("rolling_std_7", value=5000.0)
rolling_mean_28 = st.number_input("rolling_mean_28", value=27000.0)

day_of_week = st.number_input("day_of_week", value=5)
week_of_year = st.number_input("week_of_year", value=35)
month = st.number_input("month", value=9)
quarter = st.number_input("quarter", value=3)
is_weekend = st.selectbox("is_weekend", [0, 1])

if st.button("Predict"):

    payload = [
        {
            "event_date": str(event_date),
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_28": lag_28,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "rolling_mean_28": rolling_mean_28,
            "day_of_week": int(day_of_week),
            "week_of_year": int(week_of_year),
            "month": int(month),
            "quarter": int(quarter),
            "is_weekend": int(is_weekend),
        }
    ]

    response = requests.post("http://localhost:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Revenue: {result[0]['prediction']:.2f}")
    else:
        st.error(response.text)
