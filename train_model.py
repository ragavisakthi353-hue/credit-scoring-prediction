"""
Credit Scoring Prediction - Model Training Script
==================================================
This script:
1. Generates a realistic synthetic credit applicant dataset with 2,000 samples.
2. Saves the dataset to 'credit_data.csv'.
3. Builds an end-to-end Machine Learning pipeline (Preprocessing + Random Forest Classifier).
4. Trains and evaluates the model on a held-out test set.
5. Exports the trained model pipeline and evaluation metrics to the 'model/' directory.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_synthetic_credit_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for credit scoring.
    Features follow typical consumer credit risk characteristics.
    """
    print(f"[*] Generating {n_samples} synthetic credit applicant records...")

    # 1. Demographics & Employment
    age = np.random.randint(19, 70, size=n_samples)
    # Employment years bounded roughly by age - 18
    max_emp_years = np.maximum(0, age - 18)
    employment_years = np.clip(np.random.exponential(scale=5, size=n_samples), 0, max_emp_years)
    employment_years = np.round(employment_years, 1)

    # 2. Income & Housing
    # Log-normal distribution for realistic income skew
    annual_income = np.random.lognormal(mean=10.8, sigma=0.55, size=n_samples)
    annual_income = np.clip(annual_income, 18000, 220000)
    annual_income = np.round(annual_income, -2)  # round to nearest 100

    home_options = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    home_probs = [0.48, 0.40, 0.10, 0.02]
    home_ownership = np.random.choice(home_options, size=n_samples, p=home_probs)

    # 3. Loan Details
    intent_options = [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    intent_probs = [0.20, 0.18, 0.17, 0.16, 0.15, 0.14]
    loan_intent = np.random.choice(intent_options, size=n_samples, p=intent_probs)

    # Loan amount depends somewhat on income
    loan_amount = (annual_income * np.random.uniform(0.05, 0.45, size=n_samples))
    loan_amount = np.clip(loan_amount, 1000, 45000)
    loan_amount = np.round(loan_amount, -2)

    # Loan percent of income
    loan_percent_income = np.round(loan_amount / annual_income, 3)

    # Interest rate based partially on risk factors
    base_interest = 11.0 + (loan_percent_income * 12.0) - (employment_years * 0.2)
    interest_rate = np.clip(base_interest + np.random.normal(0, 2.5, size=n_samples), 5.0, 24.5)
    interest_rate = np.round(interest_rate, 2)

    # 4. Credit History
    # Credit score (FICO range 300 to 850)
    base_score = 650 + (annual_income / 10000) * 8 - (loan_percent_income * 120)
    credit_score = np.clip(base_score + np.random.normal(0, 65, size=n_samples), 300, 850)
    credit_score = np.round(credit_score).astype(int)

    # Historical default flag on file
    default_prob = np.where(credit_score < 580, 0.55, np.where(credit_score < 670, 0.22, 0.05))
    cb_person_default_on_file = np.where(
        np.random.uniform(0, 1, size=n_samples) < default_prob, "Y", "N"
    )

    # Credit history length in years (at most age - 18)
    credit_history_years = np.clip(
        np.round(np.random.uniform(1, 25, size=n_samples)), 1, np.maximum(1, age - 18)
    ).astype(int)

    # 5. Realistic Ground Truth Risk Function
    # Compute standard score representing creditworthiness
    z_score = (
        + 0.012 * (credit_score - 640)
        + 0.000025 * (annual_income - 50000)
        + 0.06 * employment_years
        - 4.2 * (loan_percent_income - 0.20)
        - 0.12 * (interest_rate - 11.0)
        - 1.4 * (cb_person_default_on_file == "Y")
        + 0.35 * (home_ownership == "OWN")
        + 0.15 * (home_ownership == "MORTGAGE")
        - 0.20 * (home_ownership == "RENT")
        + 0.02 * credit_history_years
        + np.random.logistic(loc=0, scale=0.6, size=n_samples)
    )

    # Convert to probability via sigmoid function
    prob_good_credit = 1.0 / (1.0 + np.exp(-z_score))

    # Binary outcome: 1 = Good Credit (Low Risk / Approved), 0 = Bad Credit (High Risk / Default)
    credit_status = (prob_good_credit >= 0.50).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "home_ownership": home_ownership,
        "employment_years": employment_years,
        "loan_intent": loan_intent,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "credit_score": credit_score,
        "credit_history_years": credit_history_years,
        "credit_status": credit_status,
    })

    return df


def train_and_evaluate_model():
    """
    Trains an end-to-end Machine Learning pipeline and evaluates performance.
    """
    # 1. Generate & save dataset
    df = generate_synthetic_credit_data(n_samples=2000)
    data_csv_path = "credit_data.csv"
    df.to_csv(data_csv_path, index=False)
    print(f"[+] Dataset saved successfully to '{data_csv_path}' ({len(df)} rows)")

    # Display label distribution
    good_count = int((df["credit_status"] == 1).sum())
    bad_count = int((df["credit_status"] == 0).sum())
    print(f"    - Good Credit (1): {good_count} ({good_count / len(df) * 100:.1f}%)")
    print(f"    - Bad Credit (0):  {bad_count} ({bad_count / len(df) * 100:.1f}%)")

    # 2. Features and Target separation
    X = df.drop(columns=["credit_status"])
    y = df["credit_status"]

    # Numeric and Categorical columns
    numeric_features = [
        "age",
        "annual_income",
        "employment_years",
        "loan_amount",
        "interest_rate",
        "loan_percent_income",
        "credit_score",
        "credit_history_years",
    ]
    categorical_features = [
        "home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]

    # 3. Train/Test Split (80% Train, 20% Test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[+] Train set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 4. Build Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # 5. Build Complete Model Pipeline with Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_classifier),
        ]
    )

    # 6. Train the model
    print("[*] Training Random Forest Classification Pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("[+] Model training complete!")

    # 7. Evaluate on held-out test data
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 50)
    print("       MODEL EVALUATION METRICS (TEST SET)       ")
    print("=" * 50)
    print(f" Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1-Score:  {f1:.4f}")
    print(f" ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f" True Negatives (Bad Credit):  {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f" False Negatives:              {cm[1][0]} | True Positives (Good Credit): {cm[1][1]}")
    print("=" * 50 + "\n")

    # 8. Extract Feature Importances
    cat_encoder = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    importances = model_pipeline.named_steps["classifier"].feature_importances_
    feature_importance_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in zip(all_feature_names, importances)
    ]
    feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

    print("[*] Top 5 Most Influential Features:")
    for item in feature_importance_list[:5]:
        print(f"    - {item['feature']}: {item['importance'] * 100:.2f}%")

    # 9. Save artifacts to 'model/' directory
    os.makedirs("model", exist_ok=True)

    model_filepath = os.path.join("model", "credit_model.joblib")
    joblib.dump(model_pipeline, model_filepath)
    print(f"[+] Saved trained model pipeline to '{model_filepath}'")

    metrics_payload = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "test_samples": len(y_test),
        "train_samples": len(X_train),
    }

    metrics_filepath = os.path.join("model", "metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"[+] Saved metrics to '{metrics_filepath}'")

    features_filepath = os.path.join("model", "feature_importance.json")
    with open(features_filepath, "w") as f:
        json.dump(feature_importance_list, f, indent=4)
    print(f"[+] Saved feature importances to '{features_filepath}'")

    print("\n[SUCCESS] All artifacts trained and saved ready for Streamlit!")


if __name__ == "__main__":
    train_and_evaluate_model()
