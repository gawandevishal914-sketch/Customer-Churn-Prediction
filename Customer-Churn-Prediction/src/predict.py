import joblib
from preprocess import load_and_preprocess

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Load processed dataset
df = load_and_preprocess()

# Remove target column
X = df.drop("Churn", axis=1)

# Take first customer data
sample = X.iloc[[0]]

# Predict
prediction = model.predict(sample)

# Output
if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Not Churn")