import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta
from predictions import predict_sales

# Load model & encoders
model = joblib.load("xgb_sales_quantity_model.pkl")
sku_encoder = joblib.load("sku_encoder.pkl")


# Streamlit UI
st.title("Sales Quantity Prediction")

order_date = st.date_input("Order Date")
end_date = st.date_input("End Date")
item_sku_code = st.text_input("Item SKU Code")
days_for_prediction = (end_date - order_date).days

prediction = 0

if st.button("Predict Quantity"):
    for i in range(days_for_prediction):
        order_date = order_date + timedelta(days=i)
        prediction += predict_sales(item_sku_code, order_date, model, sku_encoder)
    st.success(f"Predicted Quantity: {round(prediction)}")
