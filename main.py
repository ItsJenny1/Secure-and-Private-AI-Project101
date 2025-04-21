import pandas as pd
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
def average_stay(df):
    return df.groupby("Medical Condition")["Length of Stay"].mean().reset_index(name="Avg_Stay")

def billing_sum(df):
    return df.groupby("Medical Condition")["Billing Amount"].sum().reset_index(name="Sum_Billing")

def average_age(df):
    return df.groupby("Medical Condition")["Age"].mean().reset_index(name="Avg_Age")

def people_count(df):
    return df.groupby("Medical Condition").size().reset_index(name="Patient_Count")

# Combine results
df1 = average_stay(df)
df2 = billing_sum(df)
df3 = average_age(df)
df4 = people_count(df)

final = df1.merge(df2, on="Medical Condition").merge(df3, on="Medical Condition").merge(df4, on="Medical Condition")
print(final)

# Visual Plot
sns.scatterplot(data=final, x="Avg_Stay", y="Sum_Billing", hue="Medical Condition")
plt.title("Billing Sum vs Avg Stay by Condition (Non-DP)")
plt.xlabel("Average Stay (days)")
plt.ylabel("Total Billing ($)")
plt.show()
