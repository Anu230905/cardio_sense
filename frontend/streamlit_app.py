import pandas as pd
import requests
import streamlit as st

PREDICT_URL = "http://localhost:8000/predict"
EXPLAIN_URL = "http://localhost:8000/explain"

st.set_page_config(page_title="CardioSense AI", page_icon="🫀", layout="centered")

st.title("🫀 CardioSense AI — Heart Disease Risk (Demo)")
st.caption(
    "Educational project only — this is not a medical diagnosis. "
    "Always consult a qualified healthcare professional."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 54)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest pain type", options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"][x],
        )
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 60, 250, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 700, 246)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox("Resting ECG result", options=[0, 1, 2])

    with col2:
        thalach = st.number_input("Max heart rate achieved", 50, 250, 150)
        exang = st.selectbox("Exercise-induced angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", options=[0, 1, 2])
        ca = st.selectbox("Number of major vessels colored", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia category", options=[0, 1, 2, 3])

    submitted = st.form_submit_button("Predict risk")

if submitted:
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    try:
        pred_response = requests.post(PREDICT_URL, json=payload, timeout=10)
        pred_response.raise_for_status()
        result = pred_response.json()

        st.divider()
        prob_pct = result["risk_probability"] * 100
        if result["risk_label"] == "High risk":
            st.error(f"**{result['risk_label']}** — estimated probability: {prob_pct:.1f}%")
        else:
            st.success(f"**{result['risk_label']}** — estimated probability: {prob_pct:.1f}%")

        st.progress(min(int(prob_pct), 100))
        st.caption(result["disclaimer"])

        # --- Phase 2: SHAP explanation ---
        st.subheader("Why this result? (SHAP explanation)")
        explain_response = requests.post(EXPLAIN_URL, json=payload, timeout=10)
        explain_response.raise_for_status()
        explanation = explain_response.json()

        contrib_df = pd.DataFrame(explanation["contributions"])
        contrib_df = contrib_df.sort_values("shap_value")
        chart_df = contrib_df.set_index("label")[["shap_value"]]
        st.bar_chart(chart_df, horizontal=True)

        st.caption(
            "Bars to the right push the prediction toward higher risk; "
            "bars to the left push it toward lower risk."
        )
        with st.expander("See exact feature values and SHAP contributions"):
            st.dataframe(
                contrib_df[["label", "value", "shap_value", "direction"]],
                use_container_width=True,
                hide_index=True,
            )
        st.caption(explanation["disclaimer"])

    except requests.exceptions.ConnectionError:
        st.warning(
            "Couldn't reach the API. Make sure it's running: "
            "`uvicorn app.main:app --reload --port 8000` from the backend/ folder."
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
