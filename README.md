# 💳 Credit Scoring Prediction System

An end-to-end, beginner-friendly Machine Learning and Streamlit web application designed to evaluate loan applications and predict whether a borrower is likely to have **Good Credit** (Low Risk / Approved) or **Bad Credit** (High Risk / Default).

---

## 🌟 Key Highlights

- **Complete Machine Learning Pipeline**: Built using `scikit-learn` and `pandas`, incorporating automated preprocessing (`StandardScaler`, `OneHotEncoder`) and an optimized `RandomForestClassifier`.
- **Interactive Streamlit Web Dashboard**: Clean financial interface with real-time risk estimation, probability gauges, and instant applicant profiling.
- **Explainability & Metrics**: Confusion matrix, ROC-AUC score, classification reports, and interactive feature importance charts showing what factors drive approval decisions.
- **Realistic Synthetic Dataset**: 2,000 synthetic records modeling authentic consumer lending dynamics (FICO credit score, debt-to-income ratio, prior defaults, annual income, loan intent).
- **Zero Data Leakage**: Standardized inside a unified `sklearn.pipeline.Pipeline` that guarantees consistent preprocessing during training and real-time inference.

---

## 📂 Project Structure

```text
credit-scoring-project/
│
├── app.py                     # Streamlit web application dashboard
├── train_model.py             # Data synthesis, model training & evaluation script
├── requirements.txt           # Python library dependencies
├── README.md                  # Detailed documentation and setup guide
├── credit_data.csv            # Generated synthetic loan applicant dataset
│
└── model/                     # Trained ML model and exported evaluation metrics
    ├── credit_model.joblib    # Serialized end-to-end scikit-learn pipeline
    ├── metrics.json           # Evaluation metrics (Accuracy, ROC-AUC, F1, etc.)
    └── feature_importance.json# Ranked feature impact weights
```

---

## 📊 Dataset Schema

The dataset `credit_data.csv` represents loan applications with the following key attributes:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Applicant's age in years (19 – 70) |
| `annual_income` | Float | Gross annual income in USD ($18,000 – $220,000) |
| `home_ownership` | Categorical | Housing status (`RENT`, `MORTGAGE`, `OWN`, `OTHER`) |
| `employment_years` | Float | Years of employment at current job |
| `loan_intent` | Categorical | Purpose of loan (`EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, etc.) |
| `loan_amount` | Float | Requested loan principal in USD ($1,000 – $45,000) |
| `interest_rate` | Float | Annual interest rate percentage (5.0% – 24.5%) |
| `loan_percent_income`| Float | Ratio of loan amount to annual income (DTI indicator) |
| `cb_person_default_on_file`| Binary | Prior default record on credit bureau file (`Y` / `N`) |
| `credit_score` | Integer | Credit bureau score / FICO equivalent (300 – 850) |
| `credit_history_years`| Integer | Length of credit history on record (1 – 25 years) |
| **`credit_status` (Target)** | Binary | **`1` = Good Credit (Approved)**, **`0` = Bad Credit (High Risk)** |

---

## 🧠 Machine Learning Architecture

```
Raw Applicant Input
       │
       ▼
ColumnTransformer
  ├── Numeric Features ──► StandardScaler (Normalizes age, income, loan, credit score)
  └── Categorical Features ──► OneHotEncoder (Encodes housing, intent, default flag)
       │
       ▼
RandomForestClassifier (150 Trees, max_depth=12)
       │
       ▼
Prediction & Probability [P(Good Credit), P(Bad Credit)]
```

### Why Random Forest?
- Handles complex non-linear relationships between financial variables.
- Resilient to outliers and multicollinearity.
- Provides direct feature importance metrics.
- Ensembles multiple decision trees to mitigate overfitting.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your machine. Check with:
```bash
python --version
```

### 2. Set Up a Virtual Environment (Recommended)

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model & Generate Data
Run the training script to synthesize the dataset, train the Random Forest pipeline, and export artifacts:
```bash
python train_model.py
```
*Output will display dataset statistics, class balance, model evaluation metrics, and top feature rankings.*

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive dashboard.

---

## 🖥️ Web Application Features

1. **Interactive Credit Risk Assessor**:
   - Sliders and input selectors for all applicant characteristics.
   - Quick-load preset applicant profiles (**Prime**, **Moderate**, **Subprime**) for rapid demonstration.
   - Instant approval decision card with probability gauge.
   - Personalized financial risk analysis and advisory recommendations.

2. **Model Performance Hub**:
   - Test-set evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Confusion Matrix breakdown (False Positives vs False Negatives).
   - Interactive feature importance bar chart.

3. **Data Explorer**:
   - Interactive table viewer for `credit_data.csv`.
   - Data distributions (Credit score histogram, loan amount vs. income scatter plot).
   - Direct CSV download option.

4. **Educational ML Guide**:
   - Beginner-friendly explanations of credit scoring concepts.
   - Explanation of false negatives vs false positives in lending risk.
   - Ethical AI and fair lending principles (FCRA, ECOA compliance guidelines).

---

## 📈 Model Evaluation Summary

Typical performance on held-out test data:
- **Accuracy**: ~89% – 93%
- **ROC-AUC**: ~0.94 – 0.97
- **F1-Score**: ~0.92 – 0.95
- **Top Drivers**: FICO Credit Score, Loan-to-Income Ratio, Historical Bureau Default, and Annual Income.

---

## ⚖️ Ethical Lending & Responsible AI Note

In real-world credit risk systems, automated decisioning models must comply with fair lending laws such as the **Equal Credit Opportunity Act (ECOA)** and the **Fair Credit Reporting Act (FCRA)**. Protected attributes (race, gender, religion, marital status) are prohibited from model training, and credit institutions must provide **adverse action notices** explaining specific reasons whenever credit is denied.
