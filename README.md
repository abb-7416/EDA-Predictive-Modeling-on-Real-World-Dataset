# 📊 Telco Customer Churn — EDA & Predictive Modeling

## 🎯 Objective
Perform **Exploratory Data Analysis (EDA)** and build **Machine Learning predictive models** to classify customers likely to churn using the **Telco Customer Churn** dataset.

---

## 📦 Dataset
**Source:** [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

This dataset contains customer information for a telecom company — including demographic details, account information, and service usage — with a binary target variable **`Churn`** indicating whether the customer left the company.

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Key Features
| Column | Description |
|:--------|:-------------|
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer demographics |
| `tenure` | Number of months customer has stayed |
| `PhoneService`, `InternetService`, `OnlineSecurity`, ... | Services subscribed |
| `Contract`, `PaymentMethod`, `PaperlessBilling` | Account/contract details |
| `MonthlyCharges`, `TotalCharges` | Billing information |
| `Churn` | Target variable (`Yes` / `No`) |

---

## 🧰 Tools and Libraries
| Category | Tools Used |
|:----------|:------------|
| **Language** | Python (3.8+) |
| **Environment** | Jupyter Notebook |
| **Libraries** | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `imbalanced-learn` (for SMOTE) |

---

## 🧱 Step-by-Step Process

### 1️⃣ Data Loading and Inspection
- Imported the dataset into a Pandas DataFrame.
- Verified dataset size, structure, and column data types.
- Checked for duplicates and missing values.
- Converted `TotalCharges` from `object` to numeric and handled blank spaces.

---

### 2️⃣ Data Cleaning
- Filled missing `TotalCharges` values with median or 0.
- Removed irrelevant columns such as `customerID`.
- Ensured data consistency in categorical fields (`Yes`/`No` values).

---

### 3️⃣ Exploratory Data Analysis (EDA)
Performed both **univariate** and **bivariate** analysis:

#### 🔹 Univariate Analysis
- Visualized distributions of numeric variables (`tenure`, `MonthlyCharges`, `TotalCharges`) using histograms.
- Analyzed churn proportion (`Yes` vs `No`).

#### 🔹 Bivariate Analysis
- Compared churn rate across categories (e.g., `Contract`, `PaymentMethod`, `InternetService`).
- Boxplots for numeric variables grouped by churn.
- Heatmap of correlations among numeric variables.

#### 🔹 Insights Example
- Month-to-month contracts have higher churn.
- Longer tenure customers are less likely to churn.
- Higher `MonthlyCharges` slightly increases churn risk.

---

### 4️⃣ Feature Engineering
- Created `tenure_group` bins (e.g., `0–12`, `13–24`, …).
- Mapped `SeniorCitizen` 0/1 → `No`/`Yes`.
- Encoded categorical variables using **OneHotEncoder**.
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) with **StandardScaler** for Logistic Regression.

---

### 5️⃣ Handling Class Imbalance
- Checked class imbalance: churn rate typically ≈ 26%.
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** using `imblearn.over_sampling.SMOTE` to balance the training data.
- Alternatively, used `class_weight='balanced'` in models (for Logistic Regression and Random Forest).

---

### 6️⃣ Model Building

#### 📘 Model 1: Logistic Regression
- Used pipeline: preprocessing (scaling + encoding) → `LogisticRegression(class_weight='balanced')`.
- Tuned hyperparameters using **GridSearchCV** for `C` (regularization strength).
- Model emphasizes interpretability and linear decision boundaries.

#### 🌲 Model 2: Random Forest
- Used pipeline: preprocessing → `RandomForestClassifier(class_weight='balanced')`.
- Tuned parameters using **GridSearchCV** for:
  - `n_estimators` (100–300)
  - `max_depth` (5–20)
  - `min_samples_split`, `min_samples_leaf`
- Captures nonlinear relationships and feature interactions.

---

### 7️⃣ Model Evaluation Metrics
Evaluated models using **test data** on multiple metrics:

| Metric | Description |
|:--------|:-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Fraction of true churns among predicted churns |
| **Recall (Sensitivity)** | Fraction of actual churns correctly identified |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC AUC** | Area under ROC curve — model’s ability to distinguish classes |
| **Confusion Matrix** | True vs predicted outcomes visualization |

#### Visual Evaluations
- ROC Curve comparison (Logistic Regression vs Random Forest)
- Feature importance visualization from Random Forest

---

### 8️⃣ Model Saving
- Best-performing model (based on ROC AUC) saved using:
  ```python
  joblib.dump(best_model, 'best_churn_model.joblib')
  
### Key Observations
Churn rate: ≈ 26%
Top features influencing churn:
Contract type (Month-to-month customers churn more)
Tenure (shorter tenure = higher churn)
MonthlyCharges (higher bills correlate with churn)
InternetService type (Fiber optic customers churn more)
Model performance:
Logistic Regression AUC ≈ 0.83
Random Forest AUC ≈ 0.87
Random Forest chosen as final model (better generalization and feature insight)

**🚀 How to Run the Project**
Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
Download dataset
From Kaggle Telco Customer Churn
Place the CSV file in the same directory as the notebook.
Run notebook
Open Jupyter Notebook.
Execute all cells in eda_predictive_model.ipynb.
Observe outputs
Plots: distribution, correlation, churn patterns.
Metrics: accuracy, F1 score, ROC AUC.
Saved model file in project folder.

**📚 Learning Outcome**
By completing this project, you’ll gain hands-on experience in:
Data cleaning and preprocessing real-world data
Visualizing data distributions and relationships (EDA).
Handling categorical and numerical features in pipelines.
Managing imbalanced datasets using SMOTE or class weights.
Building and evaluating supervised ML models.
Comparing model performance using ROC AUC and F1 score.
Saving and reusing trained models for deployment.

**🏁 Summary**
This project demonstrates an end-to-end Data Science workflow:
Data → EDA → Feature Engineering → Model Building → Evaluation → Deployment-ready Model
It combines statistical understanding, Python programming, and machine learning skills, preparing you for practical analytics and AI-driven business problem solving.
