import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("xgb_sales_quantity_model.pkl")
le_sku = joblib.load("sku_encoder.pkl")
le_zone = joblib.load("zone_encoder.pkl")
le_platform = joblib.load("platform_encoder.pkl")


def predict_sales(item_sku_code, order_date, model, le_sku, le_zone, le_platform, zone, platform, selling_price):
    
    """
    A very important note: If you are adding any feature in the training, 
    place them in the same location in this function as well.
    For Example:
    As in the training the features are in the order: 
    ['Item SKU Code', 'Order Date', 'Shipping Address State', 'Channel Name', 'Selling Price']
    The same order has been followed here as you can see, first we have added
    "Order Date"
    "Item SKU Code"
    "Then splitting the Order Date into year, month, date and day"
    "Zone"
    "Platform"
    "Selling Price"
    
    """

    df = pd.DataFrame({
        "Order Date": [pd.to_datetime(order_date)],
        "Item SKU Code": [item_sku_code],
    })

    # Feature engineering
    df['order_year'] = df['Order Date'].dt.year
    df['order_month'] = df['Order Date'].dt.month
    df['order_day'] = df['Order Date'].dt.day
    df['order_dayofweek'] = df['Order Date'].dt.dayofweek

    df = df.drop(columns=["Order Date"])

    df["Zone"] = zone
    df["Platform"] = platform
    df["Selling Price"] = selling_price

    # Encode
    df["Item SKU Code"] = le_sku.transform(df["Item SKU Code"])
    df["Zone"] = le_zone.transform(df["Zone"])
    df["Platform"] = le_platform.transform(df["Platform"])

    # Predict
    prediction = model.predict(df)[0]
    return prediction

# # EXAMPLE USAGE
# pred = predict_sales("ABSORBIAPANTL30", "2025-04-07", model, le_sku)
# print(f"📦 Predicted Quantity of Sales: {round(pred)}")
