import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error

# Load data
print("Training of the Model has been started.............")
data = pd.read_excel(r"C:\Users\Satyam\Documents\Visual Studio Code\python files\nexten_brands_projects\unicommerce_oct_feb.xlsx")
zone_classification_data = pd.read_excel(r"C:\Users\Satyam\Documents\Visual Studio Code\python files\nexten_brands_projects\Indian_State_Zone_Mapping.xlsx")

data.rename(columns={'Order Date as dd/mm/yyyy hh:MM:ss': 'Order Date'}, inplace=True)
data['Order Date'] = data['Order Date'].dt.date
data['Selling Price'] = data['Selling Price'].astype(float)

data = data[['Item SKU Code', 'Order Date', 'Shipping Address State', 'Channel Name', 'Selling Price']].dropna()
data = data.groupby(['Item SKU Code', 'Order Date', 'Shipping Address State', 'Channel Name', 'Selling Price']).size().reset_index(name='Quantity')

# Performing an inner join to get the zones based on the states
merged_data = data.merge(
    zone_classification_data[['Shipping Address State', 'Zone']],
    on='Shipping Address State',
    how='left'
)

# Splitting the Channel into Location and Platform
merged_data['Platform'] = merged_data['Channel Name'].str.split('_', n=1).str[0]

merged_data['Order Date'] = pd.to_datetime(merged_data['Order Date'])
merged_data['order_year'] = merged_data['Order Date'].dt.year
merged_data['order_month'] = merged_data['Order Date'].dt.month
merged_data['order_day'] = merged_data['Order Date'].dt.day
merged_data['order_dayofweek'] = merged_data['Order Date'].dt.dayofweek  # Monday=0, Sunday=6

merged_data = merged_data.drop(['Order Date', 'Shipping Address State', 'Channel Name'], axis=1)

le_sku = LabelEncoder()
merged_data['Item SKU Code'] = le_sku.fit_transform(merged_data['Item SKU Code'])

le_zone = LabelEncoder()
merged_data['Zone'] = le_zone.fit_transform(merged_data['Zone'])

le_platform = LabelEncoder()
merged_data['Platform'] = le_platform.fit_transform(merged_data['Platform'])

x = merged_data[['Item SKU Code', 'order_year', 'order_month', 'order_day', 'order_dayofweek', 'Zone', 'Platform', 'Selling Price']]
y = merged_data['Quantity']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=23)

xgb_model = XGBRegressor(n_estimators=1000, learning_rate = 0.1, max_depth=10, random_state=42)
xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

# error calculation
error_ = mean_squared_error(y_test, xgb_predictions)
print(f"The error in this training is: {error_}")


# Save everything
joblib.dump(xgb_model, "xgb_sales_quantity_model.pkl")
joblib.dump(le_sku, "sku_encoder.pkl")
joblib.dump(le_zone, "zone_encoder.pkl")
joblib.dump(le_platform, "platform_encoder.pkl")

print("Model Training is done.")
print("✅ Model and encoders saved successfully!")