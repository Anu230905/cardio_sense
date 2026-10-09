import json
import os

import joblib
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


class HeartModel:
    """Loads the trained model once and serves predictions."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load()

    def _load(self):
        model_path = os.path.join(MODEL_DIR, "heart_model.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        features_path = os.path.join(MODEL_DIR, "feature_names.json")

        for path in (model_path, scaler_path, features_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run `python train_model.py` first "
                    f"from the backend/ folder."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            self.feature_names = json.load(f)

    def predict(self, input_dict: dict) -> dict:
        ordered = [input_dict[name] for name in self.feature_names]
        X = np.array(ordered).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        label = "High risk" if proba >= 0.5 else "Lower risk"

        return {"risk_probability": round(proba, 4), "risk_label": label}


# Singleton instance, loaded once when the app starts
heart_model = HeartModel()
