import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("healthcare_dataset.csv")
df.dropna(inplace=True)
df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"])

df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days.clip(lower=0, upper=30)

df["Billing Amount"] = df["Billing Amount"].clip(lower=0, upper=50000)

def average_stay(df):
    return df.groupby("Medical Condition")["Length of Stay"].mean().reset_index(name="Avg_Stay")

def billing_sum(df):
    return df.groupby("Medical Condition")["Billing Amount"].sum().reset_index(name="Sum_Billing")

def average_age(df):
    return df.groupby("Medical Condition")["Age"].mean().reset_index(name="Avg_Age")

def people_count(df):
    return df.groupby("Medical Condition").size().reset_index(name="Patient_Count")

df1 = average_stay(df)
df2 = billing_sum(df)
df3 = average_age(df)
df4 = people_count(df)

final = df1.merge(df2, on="Medical Condition").merge(df3, on="Medical Condition").merge(df4, on="Medical Condition")
print(final)

sns.scatterplot(data=final, x="Avg_Age", y="Sum_Billing", hue="Medical Condition")
plt.title("Sum Billing vs Avg Age by Condition (Non-DP)")
plt.xlabel("Average Age (years)")
plt.ylabel("Total Billing ($)")
plt.show()

sns.barplot(data=final, x="Medical Condition", y="Sum_Billing", color="blue")
plt.title("Non-DP: Total Billing per Medical Condition")
plt.xticks(rotation=45)
plt.ylabel("Total Billing ($)")
plt.tight_layout()
plt.show()

sns.barplot(data=final, x="Medical Condition", y="Avg_stay", color="red")
plt.title("Non-DP: Average Stay per Medical Condition")
plt.xticks(rotation=45)
plt.ylabel("Average Stay (days)")
plt.tight_layout()
plt.show()

final.to_csv("non_dp_output.csv", index=False)