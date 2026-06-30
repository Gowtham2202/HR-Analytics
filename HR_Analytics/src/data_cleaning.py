
import pandas as pd

df = pd.read_csv("D:/Elevate labs/Data Analyst/projects/New folder/HR_Analytics/data/raw/HR_Analytics-4.csv")


print("Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


df.drop_duplicates(inplace=True)


if "EmployeeCount" in df.columns:
    df.drop("EmployeeCount", axis=1, inplace=True)

if "StandardHours" in df.columns:
    df.drop("StandardHours", axis=1, inplace=True)

if "Over18" in df.columns:
    df.drop("Over18", axis=1, inplace=True)


df.to_csv("data/processed/hr_cleaned.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned Shape:", df.shape)