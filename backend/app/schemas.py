from pydantic import BaseModel, Field


class HeartInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure")
    chol: float = Field(..., ge=100, le=700, description="Serum cholesterol mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia category")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
            }
        }


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: str
    disclaimer: str = (
        "This is an educational demo, not a medical diagnosis. "
        "Consult a qualified healthcare professional."
    )


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float
    direction: str


class ExplanationResponse(BaseModel):
    base_value: float
    contributions: list[FeatureContribution]
    disclaimer: str = (
        "SHAP values show how each input pushed this specific prediction "
        "up or down relative to an average case. They explain the model's "
        "reasoning, not a medical cause-and-effect relationship."
    )
