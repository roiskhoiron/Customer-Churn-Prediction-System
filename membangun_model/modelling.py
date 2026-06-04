#!/usr/bin/env python3
import pandas as pd, mlflow, mlflow.sklearn, joblib, os, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("https://dagshub.com/roiskhoiron/Customer-Churn-EDA.mlflow")
d = os.path.dirname(os.path.abspath(__file__))
pre = os.path.join(d, "namadataset_preprocessing")
mdir = os.path.join(d, "models")

X_train = pd.read_csv(os.path.join(pre, "X_train.csv"))
X_test = pd.read_csv(os.path.join(pre, "X_test.csv"))
y_train = pd.read_csv(os.path.join(pre, "y_train.csv")).squeeze("columns")
y_test = pd.read_csv(os.path.join(pre, "y_test.csv")).squeeze("columns")
print(f"[INFO] Train {X_train.shape}, Test {X_test.shape}")

mlflow.set_experiment("SMSML_Customer_Churn_Prediction")
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"[INFO] Acc: {accuracy_score(y_test, y_pred):.4f}, "
          f"Prec: {precision_score(y_test, y_pred):.4f}, "
          f"Rec: {recall_score(y_test, y_pred):.4f}, "
          f"F1: {f1_score(y_test, y_pred):.4f}, "
          f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
    os.makedirs(mdir, exist_ok=True)
    joblib.dump(model, f"{mdir}/churn_model.pkl")
    scaler = joblib.load(os.path.join(pre, "scaler.pkl"))
    joblib.dump(scaler, f"{mdir}/scaler.pkl")
    joblib.dump(X_train.columns.tolist(), f"{mdir}/feature_names.pkl")
    print(f"[INFO] Model saved")
print("[DONE]")