import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("xgb_sales_quantity_model.pkl")
le_sku = joblib.load("sku_encoder.pkl")


def predict_sales(item_sku_code, order_date, model, le_sku):
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

    # Encode
    df["Item SKU Code"] = le_sku.transform(df["Item SKU Code"])

    # Predict
    prediction = model.predict(df)[0]
    return prediction

# # EXAMPLE USAGE
# pred = predict_sales("ABSORBIAPANTL30", "2025-04-07", model, le_sku)
# print(f"📦 Predicted Quantity of Sales: {round(pred)}")
