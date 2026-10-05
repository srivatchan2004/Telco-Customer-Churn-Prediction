# Telco Customer Churn Prediction

## 📌 Project Overview

This project demonstrates how to build an end-to-end machine learning pipeline using Python and Scikit-Learn to predict customer churn in the telecommunications industry. 

The pipeline automates:
- **Data Cleaning & Preprocessing** (missing value imputation, feature scaling, one-hot encoding)
- **ColumnTransformer** to cleanly separate numerical and categorical feature pipelines
- **Hyperparameter Tuning** with `GridSearchCV` and 5-fold cross-validation
- **Model Comparison** between **Logistic Regression** and **Random Forest Classifier**
- **Model Export** using `joblib` so the trained pipeline can immediately make predictions on new customer data

---

## 🎯 Objective

Predict whether a telecom customer is likely to cancel their service (churn) based on their contract type, tenure, payment method, monthly charges, and subscribed services.

---

## 📁 Project Structure

```
Telco-Customer-Churn-Prediction/
├── Telco-Customer-Churn.csv              # Telco Customer Churn dataset
├── telco-customer-churn-prediction.ipynb # Step-by-step Jupyter Notebook
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Git ignore rules (virtualenv, caches)
└── README.md                             # Project documentation
```

---

## 📊 Dataset

- **Total Records:** 7,043 customers
- **Features:** 20 features (demographics, services signed up for, account information)
- **Target Variable:** `Churn`
  - `0` / `No` = Customer stays
  - `1` / `Yes` = Customer leaves (~26.5% churn rate)

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** & **NumPy** for data manipulation
- **Scikit-Learn** for pipelines, preprocessing, model training, and evaluation
- **Joblib** for model serialization
- **Jupyter Notebook** for interactive exploration and execution

---

## ⚙️ Machine Learning Workflow

```
Load Dataset (Telco-Customer-Churn.csv)
      ↓
Data Cleaning (Drop customerID, fix TotalCharges data type)
      ↓
Target Encoding (Map Churn: No -> 0, Yes -> 1)
      ↓
Train-Test Split (80% Train, 20% Test with Stratification)
      ↓
Scikit-Learn Preprocessing:
  ├── Numeric Features: SimpleImputer(median) + StandardScaler()
  └── Categorical Features: SimpleImputer(most_frequent) + OneHotEncoder()
      ↓
ColumnTransformer
      ↓
Model Training & 5-Fold Cross-Validation (GridSearchCV)
  ├── 1. Logistic Regression
  └── 2. Random Forest Classifier
      ↓
Model Comparison & Evaluation (Accuracy, Precision, Recall, F1-Score)
      ↓
Export Best Pipeline (logistic_pipeline.pkl)
      ↓
Inference on New Customer Data
```

---

## 📈 Model Performance & Comparison

Both models were trained using 5-Fold Cross-Validation via `GridSearchCV` on the training set and evaluated on the held-out test set:

| Model | Test Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.55%** | **0.66** | **0.56** | **0.60** |
| **Random Forest** | **80.00%** | **0.64** | **0.53** | **0.58** |

### 🏆 Final Model Selection
**Logistic Regression** achieved the highest test accuracy (80.55%), recall (0.56), and F1-score (0.60). The complete pipeline (preprocessing + trained classifier) is saved to **`logistic_pipeline.pkl`**.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/srivatchan2004/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction
```

### 2. Set up virtual environment & install dependencies
```bash
# Optional: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Open and run the Jupyter Notebook
```bash
jupyter notebook telco-customer-churn-prediction.ipynb
```
Select **Kernel -> Restart & Run All** to run all steps from data loading to new customer prediction.

---

## 🔮 Making Predictions on New Customers

The exported pipeline (`logistic_pipeline.pkl`) contains both the data preprocessor and the classifier. You can pass raw customer data directly without any manual preprocessing:

```python
import joblib
import pandas as pd

# 1. Load the trained pipeline
pipeline = joblib.load("logistic_pipeline.pkl")

# 2. Input raw customer features
new_customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.50,
    "TotalCharges": 906.00
}])

# 3. Predict
prediction = pipeline.predict(new_customer)[0]
probability = pipeline.predict_proba(new_customer)[0][1]

print("Prediction:", "Will Churn" if prediction == 1 else "Will Stay")
print(f"Churn Probability: {probability:.2%}")
```

---

## 💡 What I Learned

- Building robust, leak-free machine learning workflows with Scikit-Learn `Pipeline` and `ColumnTransformer`.
- Handling mixed feature types (numeric and categorical) simultaneously.
- Hyperparameter tuning using `GridSearchCV` with cross-validation.
- Evaluating classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) beyond simple accuracy.
- Exporting and reloading serialized model pipelines for inference.

---

## 🔮 Future Improvements

- Experiment with gradient boosting algorithms (**XGBoost**, **LightGBM**, **CatBoost**).
- Address class imbalance using `class_weight='balanced'` or **SMOTE**.
- Build an interactive web app using **Streamlit** or **Flask** for non-technical users.

---

## 👤 Author

- **GitHub:** [@srivatchan2004](https://github.com/srivatchan2004)
