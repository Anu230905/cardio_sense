# CardioSense AI — Phase 1: Heart Disease Prediction

This is the **working foundation** of the larger CardioSense AI idea, scoped down to
something a single final-year student can actually finish and defend in an interview.

> ⚠️ Educational project only. Not a medical device. Do not use for real diagnosis.

## What's included in this phase
- Data loading + preprocessing pipeline (`backend/train_model.py`)
- A trained ML model (Logistic Regression baseline + XGBoost) with proper
  train/test split, cross-validation, and metrics (accuracy, ROC-AUC, confusion matrix)
- A FastAPI backend serving predictions (`backend/app/main.py`)
- A Streamlit frontend for a quick, working UI (`frontend/streamlit_app.py`)
- Model artifact saving/loading with `joblib`

## Folder structure
```
cardiosense-phase1/
├── backend/
│   ├── train_model.py        # trains + saves the model
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI app
│       ├── schemas.py        # request/response models
│       └── model.py          # model loading + prediction logic
├── frontend/
│   └── streamlit_app.py      # simple dashboard UI
├── data/
│   └── README.md             # where to get the dataset
└── models/                   # trained model gets saved here
```

## Setup (VS Code, step by step)

1. **Open the folder** `cardiosense-phase1` in VS Code (`File > Open Folder`).

2. **Create a virtual environment** (VS Code terminal, `Ctrl+` `` ` ``):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Get the dataset** — see `data/README.md`. Put `heart.csv` in the `data/` folder.

5. **Train the model**:
   ```bash
   cd backend
   python train_model.py
   ```
   This prints metrics and saves `models/heart_model.joblib` +
   `models/scaler.joblib` + `models/feature_names.json`.

6. **Run the API**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```
   Visit `http://localhost:8000/docs` for interactive Swagger UI.

7. **Run the frontend** (in a second terminal, venv activated):
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```

## Why Streamlit instead of React for Phase 1
React + FastAPI is the "resume-correct" combo, but it roughly doubles build time
for zero ML-learning value. Recommended path:
- Phase 1–3: Streamlit (fast iteration, you focus on ML/RAG/XAI quality)
- Phase 4–5, or as a final polish pass: rebuild the UI in React if you have time
  left. Interviewers care far more about your modeling and system design than
  your CSS.

## Roadmap for the rest of the project

### Phase 2 — SHAP explanations
- Add `shap.TreeExplainer` for the XGBoost model.
- Return per-feature contribution values from a new `/explain` endpoint.
- Render a SHAP waterfall/bar chart in Streamlit per prediction.
- Talking point for interviews: explain *why* SHAP values are additive and how
  they differ from raw feature importance.

### Phase 3 — RAG over medical guidelines
- Pick 3–5 trusted PDFs (e.g., AHA/ACC guideline summaries, WHO fact sheets —
  use publicly available patient-education material, not paywalled journals).
- Chunk with `RecursiveCharacterTextSplitter` (LangChain), ~500 tokens, 50 overlap.
- Embed with `sentence-transformers/all-MiniLM-L6-v2`, store in ChromaDB (simplest
  to run locally, no external service needed).
- Retrieval eval: manually write 10–15 test questions with known "correct" source
  chunks, measure retrieval hit rate before you plug in the LLM. This is the
  single most impressive thing you can show — most student RAG projects skip
  evaluation entirely.
- LLM call (OpenAI or Gemini) with retrieved chunks in the prompt, and always
  show the source chunk in the UI next to the answer.

### Phase 4 — One imaging model
- Pick ONE: chest X-ray (pneumonia binary classification is a well-trodden,
  well-documented dataset — good for a clean, defensible pipeline) or ECG.
- Use transfer learning (e.g., a small pretrained CNN backbone) rather than
  training from scratch — faster, better results, and it's a legitimate
  technique to talk about.
- Grad-CAM for visual explainability (the imaging equivalent of SHAP) — very
  strong resume/demo material.

### Phase 5 — Conversational assistant + PDF report
- Wrap prediction + SHAP output + retrieved guideline snippets into a single
  LLM prompt to generate a plain-language summary.
- PDF export with `reportlab` or `weasyprint` (HTML → PDF is usually less painful).
- Add a chat endpoint that keeps prediction context in the conversation so
  users can ask "why is my risk high?" as a follow-up.

## What to explicitly leave out (and say so in your resume/report)
- Multi-disease (diabetes, kidney, liver, stroke) as *simultaneous* scope —
  pick at most one more if you have spare time, after Phase 1–4 are solid.
- Auth/user accounts — nice to have, but low signal for an ML-focused project;
  only add if you finish everything else early.
- Deployment to AWS/Render — good for a demo link, but don't let infra work eat
  time you should spend on model quality and evaluation write-ups.
