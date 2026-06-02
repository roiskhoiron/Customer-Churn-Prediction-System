#!/usr/bin/env python3
"""
modelling_tuning.py
Hyperparameter tuning for Customer Churn prediction using RandomizedSearchCV.
Logs results to MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# === Paths ===
RAW_DATA = "../data/wa_customer_churn_total.csv"
MODEL_DIR = "models"

def load_and_preprocess():
    df = pd.read_csv(RAW_DATA)
    print(f"[INFO] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Preprocessing (same as modelling.py)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    le_churn = LabelEncoder()
    df['Churn'] = le_churn.fit_transform(df['Churn'])

    ordinal_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in ordinal_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    nominal_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # Feature engineering
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    bins = [0, 12, 24, 48, 72, float('inf')]
    labels = ['0-12', '13-24', '25-48', '49-72', '72+']
    df['TenureBin'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    df['TenureBin'] = LabelEncoder().fit_transform(df['TenureBin'].astype(str))
    df['SeniorPartner'] = df['SeniorCitizen'] * df['Partner']

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"[INFO] After SMOTE: {len(X_train_res)} training samples")

    return X_train_res, X_test_scaled, y_train_res, y_test, X_train.columns.tolist()

def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

    # Define parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,          # number of parameter settings sampled
        scoring='roc_auc',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    print(f"[INFO] Best parameters: {random_search.best_params_}")
    print(f"[INFO] Best ROC-AUC score: {random_search.best_score_:.4f}")

    # Evaluate best model on test set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"[INFO] Test Accuracy: {acc:.4f}")
    print(f"[INFO] Test Precision: {precision:.4f}")
    print(f"[INFO] Test Recall: {recall:.4f}")
    print(f"[INFO] Test F1: {f1:.4f}")
    print(f"[INFO] Test ROC-AUC: {auc:.4f}")

    # MLflow logging
    mlflow.set_experiment("SMSML_Customer_Churn_Tuning")
    with mlflow.start_run(run_name=f"rf_tuning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}") as run:
        # Log best parameters
        mlflow.log_params(random_search.best_params_)
        # Log metrics
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_roc_auc": auc,
            "cv_best_score": random_search.best_score_
        })
        # Log the model
        mlflow.sklearn.log_model(best_model, artifact_path="tuned_model")

        # Save locally
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, f"{MODEL_DIR}/churn_model_tuned.pkl")
        joblib.dump(feature_names, f"{MODEL_DIR}/feature_names_tuned.pkl")
        print(f"[INFO] Tuned model saved to {MODEL_DIR}/churn_model_tuned.pkl")

    print("[DONE] Hyperparameter tuning complete. Run: mlflow ui --backend-store-uri mlruns")

if __name__ == "__main__":
    main()