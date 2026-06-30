# ==================================================
# HR Analytics - Employee Attrition Prediction
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs("outputs/model", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("D:/Elevate labs/Data Analyst/projects/New folder/HR_Analytics/data/processed/hr_cleaned.csv")

print("Dataset Shape:", df.shape)

# ==================================================
# CHECK MISSING VALUES
# ==================================================

print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

# ==================================================
# HANDLE MISSING VALUES
# ==================================================

# Numerical Columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical Columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# ==================================================
# LABEL ENCODING
# ==================================================

label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    label_encoders[col] = encoder

# ==================================================
# CHECK TARGET VARIABLE
# ==================================================

print("\nAttrition Distribution:\n")
print(df["Attrition"].value_counts())

# ==================================================
# FEATURES & TARGET
# ==================================================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ==================================================
# FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# MODEL BUILDING
# ==================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==================================================
# PREDICTION
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# MODEL ACCURACY
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ==================================================
# CLASSIFICATION REPORT
# ==================================================

report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")
print(report)

with open("outputs/model/model_report.txt", "w") as file:

    file.write(f"Accuracy: {round(accuracy * 100, 2)}%\n\n")
    file.write(report)

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Attrition", "Attrition"],
    yticklabels=["No Attrition", "Attrition"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("outputs/model/confusion_matrix.png")

plt.show()

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance["Absolute_Value"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute_Value",
    ascending=False
)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

# Save Feature Importance
importance.to_csv(
    "outputs/model/feature_importance.csv",
    index=False
)

# ==================================================
# SHAP ANALYSIS
# ==================================================

print("\nGenerating SHAP Summary Plot...")

explainer = shap.LinearExplainer(model, X_train)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    show=False
)

plt.savefig(
    "outputs/model/shap_summary.png",
    bbox_inches="tight"
)

plt.show()

print("\nSHAP Analysis Completed Successfully!")
print("\nAll outputs saved inside: outputs/model/")