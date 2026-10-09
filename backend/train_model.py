"""
Train the heart disease prediction model.

Run from the backend/ folder:
    python train_model.py

Produces (in ../models/):
    heart_model.joblib       - trained XGBoost classifier
    scaler.joblib            - fitted StandardScaler
    feature_names.json       - ordered list of feature columns
    metrics.json             - held-out test metrics (for your report/resume)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = os.path.join("..", "data", "heart.csv")
MODEL_DIR = os.path.join("..", "models")

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find {path}. Download the UCI Heart Disease dataset "
            f"and place it at data/heart.csv — see data/README.md."
        )
    df = pd.read_csv(path)
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_cols}. "
            f"Check data/README.md for the expected schema, or edit "
            f"FEATURE_COLUMNS in this script to match your CSV."
        )
    return df


def main():
    df = load_data(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    # UCI target is sometimes 0-4 severity; binarize to disease/no-disease
    y = (y > 0).astype(int)

    print(f"Rows: {len(df)} | Class balance:\n{y.value_counts(normalize=True)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Baseline: Logistic Regression (interpretable, good sanity check) ---
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    lr_preds = log_reg.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression test accuracy: {lr_acc:.3f}")

    # --- Main model: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    # 5-fold cross-validation on the training set — report this, not just a
    # single train/test split, since the dataset is small (~300 rows).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC (train set): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    xgb.fit(X_train_scaled, y_train)
    xgb_preds = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

    test_acc = accuracy_score(y_test, xgb_preds)
    test_auc = roc_auc_score(y_test, xgb_proba)
    cm = confusion_matrix(y_test, xgb_preds).tolist()
    report = classification_report(y_test, xgb_preds, output_dict=True)

    print(f"\nXGBoost held-out test accuracy: {test_acc:.3f}")
    print(f"XGBoost held-out test ROC-AUC:  {test_auc:.3f}")
    print(f"Confusion matrix: {cm}")

    # --- Save artifacts ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "heart_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metrics = {
        "logistic_regression_baseline_accuracy": lr_acc,
        "xgboost_cv_roc_auc_mean": float(cv_scores.mean()),
        "xgboost_cv_roc_auc_std": float(cv_scores.std()),
        "xgboost_test_accuracy": test_acc,
        "xgboost_test_roc_auc": test_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + scaler + metrics to {MODEL_DIR}/")
    print("Keep metrics.json — you'll want these exact numbers for your resume/report.")


if __name__ == "__main__":
    main()
