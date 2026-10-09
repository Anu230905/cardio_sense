from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.explainer import heart_explainer
from app.model import heart_model
from app.schemas import ExplanationResponse, HeartInput, PredictionResponse

app = FastAPI(
    title="CardioSense AI — Phase 1",
    description="Heart disease risk prediction API (educational demo).",
    version="0.1.0",
)

# Allow the Streamlit frontend (or a future React app) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartInput):
    try:
        result = heart_model.predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: HeartInput, top_n: int = 5):
    try:
        result = heart_explainer.explain(payload.model_dump(), top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "base_value": result["base_value"],
        "contributions": result["contributions"],
    }
