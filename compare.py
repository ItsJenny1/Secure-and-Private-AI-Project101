import pandas as pd

non_dp = pd.read_csv("non_dp_output.csv")
dp = pd.read_csv("dp_output.csv")

merged = non_dp.merge(dp, left_on="Medical Condition", right_on="Condition")

merged["% Diff Avg Stay"] = ((merged["DP_Avg_Stay"] - merged["Avg_Stay"]).abs() / merged["Avg_Stay"]) * 100
merged["% Diff Sum Billing"] = ((merged["DP_Sum_Billing"] - merged["Sum_Billing"]).abs() / merged["Sum_Billing"]) * 100
merged["% Diff Avg Age"] = ((merged["DP_Avg_Age"] - merged["Avg_Age"]).abs() / merged["Avg_Age"]) * 100

comparison = merged[[
    "Condition", "Avg_Stay", "DP_Avg_Stay", "% Diff Avg Stay",
    "Sum_Billing", "DP_Sum_Billing", "% Diff Sum Billing",
    "Avg_Age", "DP_Avg_Age", "% Diff Avg Age"
]]

print(comparison.round(2))
comparison.to_csv("dp_percent_difference_report.csv", index=False)
