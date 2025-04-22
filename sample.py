import pandas as pd
import pydp as dp  
from pydp.algorithms.laplacian import BoundedMean, BoundedSum
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("healthcare_dataset.csv")
df.dropna(inplace=True)
df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"])

df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days.clip(lower=0, upper=30)

df["Billing Amount"] = df["Billing Amount"].clip(lower=0, upper=50000)

def dpaverage_stay(df, epsilon=0.1):
    results = []
    for condition in df["Medical Condition"].unique():
        stays = df[df["Medical Condition"] == condition]["Length of Stay"].tolist()
        dp_mean = BoundedMean(epsilon=epsilon, lower_bound=0, upper_bound=30).quick_result(stays)  # Updated syntax
        results.append({"Condition": condition, "DP_Avg_Stay": dp_mean})
    return pd.DataFrame(results)

def dpbilling_sum(df, epsilon=0.1):
    results = []
    for condition in df["Medical Condition"].unique():
        bills = df[df["Medical Condition"] == condition]["Billing Amount"].astype(int).tolist()  # ✅ Convert to int
        dp_sum = BoundedSum(epsilon=epsilon, lower_bound=0, upper_bound=50000).quick_result(bills)
        results.append({"Condition": condition, "DP_Sum_Billing": dp_sum})
    return pd.DataFrame(results)


def dpaverage_age(df, epsilon=0.1):
    results = []
    for condition in df["Medical Condition"].unique():
        ages = df[df["Medical Condition"] == condition]["Age"].tolist()
        dp_mean = BoundedMean(epsilon=epsilon, lower_bound=0, upper_bound=100).quick_result(ages)  # Updated syntax
        results.append({"Condition": condition, "DP_Avg_Age": dp_mean})
    return pd.DataFrame(results)

def dppeople_count(df):
    results = []
    for condition in df["Medical Condition"].unique():
        count = len(df[df["Medical Condition"] == condition])
        results.append({"Condition": condition, "DP_Patient_Count": count})
    return pd.DataFrame(results)

df1 = dpaverage_stay(df)
df2 = dpbilling_sum(df)
df3 = dpaverage_age(df)
df4 = dppeople_count(df)

final = df1.merge(df2, on="Condition").merge(df3, on="Condition").merge(df4, on="Condition")
final.sort_values(by="Condition", inplace=True)
print(final)

sns.scatterplot(data=final, x="DP_Avg_Age", y="DP_Sum_Billing", hue="Condition")
plt.title("DP Sum Billing vs Avg Age by Condition")
plt.xlabel("DP Average Age (years)")
plt.ylabel("DP Total Billing ($)")
plt.show()

sns.barplot(data=final, x="Condition", y="DP_Sum_Billing", color="orange")
plt.title("DP: Total Billing per Medical Condition")
plt.xticks(rotation=45)
plt.ylabel("DP Total Billing ($)")
plt.tight_layout()
plt.show()

sns.barplot(data=final, x="Condition", y="DP_Avg_Stay", color="coral")
plt.title("DP: Average Stay per Medical Condition")
plt.xticks(rotation=45)
plt.ylabel("DP Average Stay (days)")
plt.tight_layout()
plt.show()
