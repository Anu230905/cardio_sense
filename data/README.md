# Dataset

Use the **UCI Heart Disease dataset** (Cleveland subset), widely used and
well-documented — good for a resume project because you can clearly explain
every feature.

Two easy sources (either works, same underlying data, sometimes different column names):

1. Kaggle: search "Heart Disease UCI" (e.g. the dataset by `ronitf` or `redwankarimsony`)
2. UCI ML Repository: https://archive.ics.uci.edu/dataset/45/heart+disease

Download and place the CSV here as `data/heart.csv`.

## Expected columns
The training script expects these columns (standard UCI naming). If your CSV
uses different names, either rename them or adjust `FEATURE_COLUMNS` in
`backend/train_model.py`:

| column | meaning |
|---|---|
| age | age in years |
| sex | 1 = male, 0 = female |
| cp | chest pain type (0-3) |
| trestbps | resting blood pressure |
| chol | serum cholesterol mg/dl |
| fbs | fasting blood sugar > 120 mg/dl (1/0) |
| restecg | resting ECG results (0-2) |
| thalach | max heart rate achieved |
| exang | exercise-induced angina (1/0) |
| oldpeak | ST depression induced by exercise |
| slope | slope of peak exercise ST segment |
| ca | number of major vessels colored by fluoroscopy |
| thal | thalassemia (categorical) |
| target | 1 = disease present, 0 = no disease |

Do a quick `df.info()` and `df['target'].value_counts()` before training —
know your class balance, it's a common interview question.
