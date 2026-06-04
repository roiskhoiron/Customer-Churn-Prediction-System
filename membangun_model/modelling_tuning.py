#!/usr/bin/env python3
import pandas as pd, mlflow, mlflow.sklearn, joblib, os, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

param_dist = {'n_estimators': [100,200,300,500], 'max_depth': [5,10,15,20,None],
    'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,4],
    'max_features': ['sqrt','log2',None], 'bootstrap': [True,False]}
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_dist, n_iter=20, scoring='roc_auc', cv=3, verbose=2, random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)
print(f"[INFO] Best params: {rs.best_params_}")

best = rs.best_estimator_
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:,1]

mlflow.set_experiment("SMSML_Customer_Churn_Tuning")
mlflow.sklearn.autolog()
with mlflow.start_run() as run:
    mlflow.log_params(rs.best_params_)
    mlflow.log_metrics({"test_accuracy": accuracy_score(y_test,y_pred),
        "test_precision": precision_score(y_test,y_pred),
        "test_recall": recall_score(y_test,y_pred),
        "test_f1": f1_score(y_test,y_pred),
        "test_roc_auc": roc_auc_score(y_test,y_proba),
        "cv_best_score": rs.best_score_})
    mlflow.sklearn.log_model(best, artifact_path="tuned_model")
    os.makedirs(mdir, exist_ok=True)
    joblib.dump(best, f"{mdir}/churn_model_tuned.pkl")
    print("[INFO] Tuned model saved")
print("[DONE]")
