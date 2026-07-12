# Task 1: Loan Approval Prediction using Machine Learning

## Project Overview

This project predicts whether a loan application should be approved based on applicant information using supervised machine learning techniques.

## Objective

Develop a machine learning model for loan approval prediction with a focus on:

- Data preprocessing
- Missing value handling
- Encoding categorical variables
- Feature scaling
- Handling class imbalance using SMOTE
- Model comparison
- Business interpretation

## Dataset

- Dataset: Loan Approval Prediction Case Study
- Total Records: 614
- Total Features: 13
- Target Variable: Loan_Status

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

## Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Missing Value Handling
4. Feature Encoding
5. Feature Scaling
6. Train-Test Split
7. SMOTE
8. Model Building
9. Model Evaluation
10. Business Insights

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest

## Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|--------|---------:|---------:|---------:|
| Logistic Regression | 0.84 | 0.89 | 0.87 |
| Random Forest | 0.80 | 0.86 | 0.77 |
| Decision Tree | 0.73 | 0.80 | 0.70 |

## Best Model

**Logistic Regression** achieved the best overall performance and was selected as the final model.

## Project Structure

```text
Task-1-Loan-Approval-Prediction/
│
├── Loan_Approval_Prediction.ipynb
├── loan_prediction.csv
└── README.md
```

## Future Improvements

- Hyperparameter tuning
- XGBoost implementation
- Streamlit deployment
- SHAP-based model explainability

## Author

**Dhruv Kumar**
