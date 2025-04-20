
import pandas as pd
import pydp as dp  # Import PyDP
from pydp.algorithms.laplacian import BoundedMean, BoundedSum
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")
df.dropna(inplace=True)
# Remove direct identifiers
df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"])

# Convert dates and calculate Length of Stay
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days.clip(lower=0, upper=30)

# Clip Billing Amount
df["Billing Amount"] = df["Billing Amount"].clip(lower=0, upper=50000)

# Functions
def get_dp_avg_stay_by_condition(df, epsilon=1.0):
    results = []
    for condition in df["Medical Condition"].unique():
        stays = df[df["Medical Condition"] == condition]["Length of Stay"].tolist()
        dp_mean = BoundedMean(epsilon=epsilon, lower_bound=0, upper_bound=30).quick_result(stays)  # Updated syntax
        results.append({"Condition": condition, "DP_Avg_Stay": dp_mean})
    return pd.DataFrame(results)

def get_dp_sum_billing_by_condition(df, epsilon=1.0):
    results = []
    for condition in df["Medical Condition"].unique():
        bills = df[df["Medical Condition"] == condition]["Billing Amount"].astype(int).tolist()  # ✅ Convert to int
        dp_sum = BoundedSum(epsilon=epsilon, lower_bound=0, upper_bound=50000).quick_result(bills)
        results.append({"Condition": condition, "DP_Sum_Billing": dp_sum})
    return pd.DataFrame(results)


def get_dp_avg_age_by_condition(df, epsilon=1.0):
    results = []
    for condition in df["Medical Condition"].unique():
        ages = df[df["Medical Condition"] == condition]["Age"].tolist()
        dp_mean = BoundedMean(epsilon=epsilon, lower_bound=0, upper_bound=100).quick_result(ages)  # Updated syntax
        results.append({"Condition": condition, "DP_Avg_Age": dp_mean})
    return pd.DataFrame(results)

def get_dp_patient_count_by_condition(df):
    results = []
    for condition in df["Medical Condition"].unique():
        count = len(df[df["Medical Condition"] == condition])
        results.append({"Condition": condition, "DP_Patient_Count": count})
    return pd.DataFrame(results)

# Combine results
df1 = get_dp_avg_stay_by_condition(df)
df2 = get_dp_sum_billing_by_condition(df)
df3 = get_dp_avg_age_by_condition(df)
df4 = get_dp_patient_count_by_condition(df)

final = df1.merge(df2, on="Condition").merge(df3, on="Condition").merge(df4, on="Condition")
print(final)

# Optional Plot
sns.scatterplot(data=final, x="DP_Avg_Stay", y="DP_Sum_Billing", hue="Condition")
plt.title("DP Sum Billing vs Avg Stay by Condition")
plt.xlabel("DP Average Stay (days)")
plt.ylabel("DP Total Billing ($)")
plt.show()
