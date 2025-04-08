import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

# Load data
print("Training of the Model has been started.............")
data = pd.read_excel(r"unicommerce_oct_feb.xlsx")


data.rename(columns={'Order Date as dd/mm/yyyy hh:MM:ss': 'Order Date'}, inplace=True)
data['Order Date'] = data['Order Date'].dt.date

data = data[['Item SKU Code', 'Selling Price', 'Order Date', 'Facility']].dropna()
data = data.groupby(['Item SKU Code', 'Order Date']).size().reset_index(name='Quantity')

# Data type conversion
data['Item SKU Code'] = data['Item SKU Code'].astype(str)
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Quantity'] = data['Quantity'].astype(float)



# Feature Engineering
data['order_year'] = data['Order Date'].dt.year
data['order_month'] = data['Order Date'].dt.month
data['order_day'] = data['Order Date'].dt.day
data['order_dayofweek'] = data['Order Date'].dt.dayofweek

# Drop original date columns
data = data.drop(columns=["Order Date"])

# Label encode SKU and FC
le_sku = LabelEncoder()

data["Item SKU Code"] = le_sku.fit_transform(data["Item SKU Code"])

# Split data
X = data.drop(columns=["Quantity"])
y = data["Quantity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Train model
model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "xgb_sales_quantity_model.pkl")
joblib.dump(le_sku, "sku_encoder.pkl")

print("Model Training is done.")
print("✅ Model and encoders saved successfully!")