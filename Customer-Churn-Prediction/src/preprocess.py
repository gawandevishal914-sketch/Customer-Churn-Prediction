import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():

    # Load dataset
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Remove customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Convert ALL object columns into string first
    object_columns = df.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()

    for column in object_columns:
        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

    return df