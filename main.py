import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta
from predictions import predict_sales

# Load model and encoders
model = joblib.load("xgb_sales_quantity_model.pkl")
le_sku = joblib.load("sku_encoder.pkl")
le_zone = joblib.load("zone_encoder.pkl")
le_platform = joblib.load("platform_encoder.pkl")

zone = ['East', 'West', 'North', 'South']
platform = ['FLIPKART', 'Meesho', 'NEXTEN']


# Streamlit UI
st.title("Sales Quantity Prediction")

# Note for the users
st.write("This application predicts the sales of a particular SKU (SKU from the Unicommerce) in the upcoming days (between the start and end date).")

order_date = st.date_input("Order Date")
end_date = st.date_input("End Date")
item_sku_code = st.text_input("Item SKU Code")
selling_price = st.number_input("Selling Price (used in that period).")
days_for_prediction = (end_date - order_date).days

if days_for_prediction == 0: days_for_prediction = 1

predictions = 0

if st.button("Predict Quantity"):
    
    for i in zone:
        for j in platform:
            # order_date = pd.to_datetime("2024-10-01")
            order_date = order_date
            for k in range(days_for_prediction):

                order_date = order_date + timedelta(days=k)
                predictions += predict_sales(item_sku_code, order_date, model, le_sku, le_zone, le_platform, i, j, selling_price)
            
            st.success(f"Prediction for {item_sku_code} from {i} zone from {j} platform is {round(predictions)}")
    
