import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("models/churn_model.pkl")

print("=== Customer Churn Prediction ===")

# Important inputs
tenure = int(input("Tenure Months: "))
monthly_charges = float(input("Monthly Charges: "))
total_charges = float(input("Total Charges: "))

# Create 19 features with default value 0
features = np.zeros(19)

# Set important values
features[4] = tenure
features[17] = monthly_charges
features[18] = total_charges

# Convert to DataFrame
data = pd.DataFrame([features])

# Prediction
prediction = model.predict(data)

# Output
if prediction[0] == 1:
    print("\nCustomer Will Churn")
else:
    print("\nCustomer Will Not Churn")