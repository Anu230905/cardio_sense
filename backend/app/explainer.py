"""
SHAP-based explainability for the heart disease model.

Why TreeExplainer: our model is XGBoost, and TreeExplainer computes exact
SHAP values for tree ensembles quickly (no sampling/approximation needed,
unlike KernelExplainer which you'd need for a black-box model).
"""

import json
import os

import joblib
import numpy as np
import shap

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# Human-readable labels so the API/UI don't show raw column names like "cp"
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "Major vessels colored",
    "thal": "Thalassemia category",
}


class HeartExplainer:
    """Loads the model once and computes per-prediction SHAP contributions."""

    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        features_path = os.path.join(MODEL_DIR, "feature_names.json")

        for path in (model_path, scaler_path, features_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run `python train_model.py` first."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            self.feature_names = json.load(f)

        # TreeExplainer reads the model's internal structure directly -
        # no background dataset required, unlike KernelExplainer.
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, input_dict: dict, top_n: int = 5) -> dict:
        ordered = [input_dict[name] for name in self.feature_names]
        X = np.array(ordered).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # shap_values: one row per sample, one column per feature.
        # For binary XGBoost classifiers, positive SHAP = pushes toward
        # "disease present" (class 1); negative = pushes toward "no disease".
        shap_values = self.explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            # Older SHAP/XGBoost combos return a list per class; take class 1
            row = shap_values[1][0]
        else:
            row = shap_values[0]

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]

        contributions = []
        for name, value, shap_val in zip(self.feature_names, ordered, row):
            contributions.append({
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": value,
                "shap_value": round(float(shap_val), 4),
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
            })

        # Sort by absolute impact, largest first
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "base_value": round(float(base_value), 4),
            "contributions": contributions[:top_n],
            "all_contributions": contributions,
        }


heart_explainer = HeartExplainer()
