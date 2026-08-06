# 🏦 Loan Approval Prediction

> **Supervised Machine Learning Pipeline** — Predict loan approval from borrower features with a full preprocessing pipeline, class-imbalance handling (SMOTE), multi-model comparison, and an interactive Streamlit dashboard.

---

## 🎯 Goal

Build a supervised ML model to predict loan approval using applicant/borrower features.  
Focus areas: **data preprocessing**, **handling class imbalance**, **model evaluation**, and **business-oriented interpretation**.

---

## 📋 Requirements Coverage

| Requirement | Status | Implementation |
|---|---|---|
| Data Preprocessing — missing values | ✅ | Median imputation (numerical) + Mode imputation (categorical) |
| Data Preprocessing — categorical encoding | ✅ | `OneHotEncoder` via `ColumnTransformer` |
| Data Preprocessing — feature scaling | ✅ | `StandardScaler` for all numerical columns |
| Handle class imbalance | ✅ | **SMOTE** (Synthetic Minority Over-sampling Technique) on training set only |
| Compare multiple models | ✅ | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Report Precision, Recall, F1, ROC-AUC | ✅ | Full metrics table + ROC curve plot |
| Business-oriented interpretation | ✅ | Business Insights, Deployment Recommendation, Conclusion sections |

---

## 📁 Project Structure

```
Loan_Approval/
│
├── Loan_Approval_Predicition.ipynb   # Main analysis notebook (107 cells, fully executed)
├── loan_prediction.csv               # Raw dataset (614 records, 13 features)
├── cleaned.csv                       # Preprocessed dataset (exported by notebook)
├── app.py                            # Streamlit interactive dashboard
├── requirements.txt                  # Python package dependencies
├── README.md                         # Project documentation (this file)
│
├── models/
│   ├── loan_approval_models.pkl      # Full model bundle (all 4 models + preprocessor)
│   └── best_loan_model.pkl          # Best model only (Random Forest)
│
└── images/
    ├── target_distribution.png
    ├── missing_values_heatmap.png
    ├── numerical_distributions.png
    ├── correlation_heatmap.png
    ├── credit_history_impact.png
    ├── property_area_impact.png
    ├── smote_class_balance.png
    ├── roc_curves_all_models.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| Source | Loan Prediction Dataset (Analytics Vidhya / Kaggle) |
| Records | 614 applicants |
| Features | 13 (12 input + 1 target) |
| Target | `Loan_Status` — Y (Approved) / N (Rejected) |
| Class ratio | ~68% Approved, ~32% Rejected |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Categorical | 0 / 1 / 2 / 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numerical | Primary applicant's monthly income |
| `CoapplicantIncome` | Numerical | Co-applicant's monthly income |
| `LoanAmount` | Numerical | Requested loan amount (×$1,000) |
| `Loan_Amount_Term` | Numerical | Loan repayment term in months |
| `Credit_History` | Numerical | 1 = Good credit / 0 = Bad credit |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |

---

## 🔬 Notebook Pipeline (`Loan_Approval_Predicition.ipynb`)

The notebook contains **107 cells** organized in the following sections:

```
1.  Problem Statement
2.  Library Imports
3.  Dataset Loading & Overview
4.  Exploratory Data Analysis (EDA)
     - Target Variable Distribution
     - Missing Value Analysis
     - Numerical Feature Distributions
     - Outlier Detection (Boxplots)
     - Categorical Feature Analysis
     - Correlation Heatmap
5.  Feature vs. Target Relationship
6.  EDA Summary & Business Insights
7.  Data Preprocessing
     - Drop Loan_ID
     - Missing Value Imputation
     - Feature Selection
     - Target Encoding (Y→1, N→0)
     - Train-Test Split (80/20, stratified)
     - Feature Encoding (OneHotEncoder)
     - Feature Scaling (StandardScaler)
8.  Handling Class Imbalance (SMOTE)
9.  Model Building & Comparison
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - (Extended with Gradient Boosting)
10. Model Evaluation
     - Classification Report
     - Confusion Matrix
     - ROC Curve
     - Feature Importance
11. Final Model Selection
12. Threshold Analysis (0.50 / 0.60 / 0.70)
13. Model Saving (models/ directory)
14. Business Insights
15. Deployment Recommendation
16. Conclusion & Future Work
```

---

## 🤖 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | 0.829 | 0.864 | 0.894 | **0.879** | 0.798 |
| Logistic Regression | 0.813 | 0.852 | 0.882 | 0.867 | **0.873** |
| Gradient Boosting | 0.772 | 0.813 | 0.871 | 0.841 | 0.779 |
| Decision Tree | 0.724 | 0.831 | 0.753 | 0.790 | 0.705 |

> **Selected Model**: `Random Forest` — best F1 Score (0.879). Logistic Regression has the highest ROC-AUC (0.873) and is preferred when interpretability is critical.

---

## 💡 Key Business Insights

1. **Credit History is the #1 predictor** — applicants with `Credit_History = 1` (good credit) have dramatically higher approval probabilities.  
2. **Higher combined income** → lower risk → higher approval likelihood.  
3. **Semiurban properties** show higher historical approval rates vs. Rural.  
4. **Random Forest** generalizes best across applicant profiles due to its ensemble nature.  
5. **SMOTE prevents under-detection** of creditworthy minority-class applicants — improving recall on the "Rejected" class from ~65% to ~89%.

---

## 🚀 Running the Streamlit App

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Rebuild Models & Images

```bash
python rebuild_bundle.py
```
> Only needed if `models/loan_approval_models.pkl` is missing.

### 3. Launch Dashboard

```bash
streamlit run app.py
```

The app runs at `http://localhost:8501` with four interactive tabs:

| Tab | Description |
|---|---|
| 🎯 **Eligibility Predictor** | Real-time loan decision with approval probability, financial ratios, risk factors, and underwriting recommendations |
| 📊 **Model Performance** | Benchmark comparison table, ROC curves, confusion matrix, feature importance |
| 📈 **Data Insights** | EDA plots — class distribution, missing value heatmap, feature distributions, correlation matrix |
| 📁 **Batch Prediction** | Upload a CSV of applicants to score multiple records at once |

### Sidebar Features
- Quick-load sample profiles (Strong / Borderline / Weak)
- Model selector (choose between 4 trained classifiers)
- Full form with demographic + financial inputs

---

## 🧪 Quick Batch Prediction (CSV)

Prepare a CSV with these exact columns:

```
Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area
```

Upload via the **Batch Prediction** tab and download results as CSV.

---

## 📦 Model Bundle Contents (`models/loan_approval_models.pkl`)

```python
{
  "models":              dict of 4 fitted classifiers,
  "best_model_name":     "Random Forest",
  "best_model":          fitted RandomForestClassifier,
  "preprocessor":        fitted ColumnTransformer (scaler + encoder),
  "feature_names":       list of preprocessed feature names,
  "results_df":          model benchmark DataFrame,
  "deployment_threshold": 0.50
}
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` |
| Class Imbalance | `imbalanced-learn` (SMOTE) |
| Visualization | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Notebook | `jupyter`, `nbconvert` |

---

## 📌 Future Work

- Hyperparameter tuning with `GridSearchCV` / `Optuna`
- XGBoost / LightGBM evaluation
- SHAP values for explainability
- Deployment via Docker + FastAPI
- Real-time model monitoring dashboard

---

*Internship Project — Loan Approval Prediction using Supervised Machine Learning*

