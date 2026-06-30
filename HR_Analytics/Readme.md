HR Analytics - Predict Employee Attrition

📌 Project Overview

Employee attrition is a major challenge for organizations because high turnover increases recruitment costs, reduces productivity, and affects employee morale. This project uses data analytics and machine learning techniques to identify the factors influencing employee attrition and predict employees who are likely to leave the organization.

The project combines Python, Machine Learning, SHAP Explainability, and Power BI to provide both predictive and business insights for decision-makers.

---

🎯 Objective

The main objectives of this project are:

* Analyze employee data to identify the major causes of attrition.
* Perform Exploratory Data Analysis (EDA) to understand employee behavior.
* Build a classification model to predict employee attrition.
* Explain model predictions using SHAP values.
* Create an interactive Power BI dashboard for business users.
* Provide recommendations to reduce employee turnover.

---

🛠️ Tools & Technologies

| Tool         | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Data Cleaning & Machine Learning |
| Pandas       | Data Manipulation                |
| NumPy        | Numerical Computations           |
| Matplotlib   | Data Visualization               |
| Seaborn      | Exploratory Data Analysis        |
| Scikit-Learn | Logistic Regression Model        |
| SHAP         | Model Explainability             |
| Power BI     | Interactive Dashboard            |

---


 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Employee Attrition Distribution
* Department-wise Attrition Analysis
* Salary Band Analysis
* Promotion vs Attrition Analysis
* Overtime vs Attrition Analysis
* Correlation Heatmap

Key Insights

* Employees working overtime showed higher attrition rates.
* Employees with fewer promotions were more likely to leave.
* Monthly income had a significant impact on retention.
* Certain departments experienced higher employee turnover.

---

🤖 Machine Learning Model

 Model Used

* Logistic Regression


 Data Preprocessing

* Missing Value Handling
* Label Encoding
* Feature Scaling using StandardScaler
* Train-Test Split (80:20)


 Model Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the Logistic Regression model.

The SHAP analysis helps identify:

* Features that increase attrition probability.
* Features that decrease attrition probability.
* The overall importance of employee attributes in prediction.

This improves transparency and helps HR teams make informed decisions.

---

📈 Power BI Dashboard

The Power BI dashboard provides interactive visualizations including:

* Overall Attrition Rate
* Department-wise Employee Analysis
* Salary Distribution
* Promotion Insights
* Overtime Analysis
* Employee Demographics
* Business KPIs

The dashboard enables HR managers to explore employee trends and identify potential retention issues.

---

🚀 How to Run the Project

Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

 Step 2: Run Data Cleaning

```bash
python src/01_data_cleaning.py
```

 Step 3: Run Exploratory Data Analysis

```bash
python src/02_eda.py
```

Step 4: Run Machine Learning Model

```bash
python src/03_attrition_prediction.py
```

---

📋 Deliverables

✔ Power BI Dashboard

✔ Exploratory Data Analysis Report

✔ Logistic Regression Model

✔ Accuracy Report

✔ Classification Report

✔ Confusion Matrix

✔ SHAP Value Analysis

✔ Attrition Prevention Recommendations

---

 💡 Attrition Prevention Suggestions

Based on the analysis, the following recommendations are proposed:

* Improve employee work-life balance policies.
* Reduce excessive overtime workloads.
* Increase promotion and career growth opportunities.
* Review compensation structures for lower-income employees.
* Develop employee engagement programs.
* Strengthen retention strategies for high-risk departments.
* Provide regular training and development opportunities.

---

