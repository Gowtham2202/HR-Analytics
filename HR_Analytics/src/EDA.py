
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/hr_cleaned.csv")


import os
os.makedirs("outputs/eda", exist_ok=True)


plt.figure(figsize=(6, 5))

sns.countplot(data=df, x="Attrition")

plt.title("Employee Attrition Distribution")

plt.savefig("outputs/eda/attrition_distribution.png")

plt.show()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)

plt.xticks(rotation=20)

plt.title("Department-wise Attrition")

plt.savefig("outputs/eda/department_attrition.png")

plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Attrition")

plt.savefig("outputs/eda/salary_attrition.png")

plt.show()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="YearsSinceLastPromotion",
    hue="Attrition"
)

plt.title("Promotion vs Attrition")

plt.savefig("outputs/eda/promotion_attrition.png")

plt.show()


plt.figure(figsize=(6, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("OverTime vs Attrition")

plt.savefig("outputs/eda/overtime_attrition.png")

plt.show()


numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(18, 12))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("outputs/eda/correlation_heatmap.png")

plt.show()

print("\nEDA Completed Successfully!")