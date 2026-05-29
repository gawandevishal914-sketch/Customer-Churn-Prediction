import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Churn count graph
sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")

plt.show()