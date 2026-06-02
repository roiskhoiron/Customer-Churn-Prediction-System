#!/usr/bin/env python3
"""
modelling.py
Train Customer Churn prediction model with MLflow tracking.
Loads raw data, performs preprocessing (including SMOTE), trains model, logs to MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

if not os.getenv('MLFLOW_TRACKING_URI'):
    mlflow.set_tracking_uri("https://dagshub.com/roiskhoiron/Customer-Churn-Prediction-System.mlflow")

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(SCRIPT_DIR, "..", "data", "wa_customer_churn_total.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# === Load Raw Data ===
df = pd.read_csv(RAW_DATA)
print(f"[INFO] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

# === Preprocessing ===
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

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

# Scale & SMOTE
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"[INFO] After SMOTE: {len(X_train_res)} training samples")

# === MLflow Experiment ===

# Tracking URI will be set by dagshub.init(mlflow=True) above
mlflow.set_experiment("SMSML_Customer_Churn_Prediction")

with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"[INFO] Accuracy: {acc:.4f}")
    print(f"[INFO] Precision: {precision:.4f}")
    print(f"[INFO] Recall: {recall:.4f}")
    print(f"[INFO] F1: {f1:.4f}")
    print(f"[INFO] ROC-AUC: {auc:.4f}")

    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "class_weight": "balanced"
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })

    mlflow.sklearn.log_model(model, artifact_path="model")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/churn_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(X_train.columns.tolist(), f"{MODEL_DIR}/feature_names.pkl")
    print(f"[INFO] Model saved to {MODEL_DIR}/churn_model.pkl")

print("[DONE] Modelling complete. Run: mlflow ui --backend-store-uri mlruns")
