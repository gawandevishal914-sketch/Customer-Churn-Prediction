import pandas as pd

def load_and_preprocess():

    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # remove ID column
    df.drop("customerID", axis=1, inplace=True)

    # fix numeric column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # target convert (Yes/No -> 1/0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ONE HOT ENCODING for all categorical features
    df = pd.get_dummies(df)

    return df