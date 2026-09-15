import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk | Heart Disease Risk Prediction",
    page_icon="assets/favicon.png" if False else "🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme: "clinical monitor" — dark charcoal canvas, phosphor-green primary
# signal (evokes an ECG readout), warm coral for elevated risk. IBM Plex for
# a technical, instrument-panel feel; the mono face is reserved for numbers.
# ----------------------------------------------------------------------------
PRIMARY = "#00E5A0"       # phosphor green — normal / healthy signal
RISK = "#FF6B6B"          # coral red — elevated risk signal
BG = "#0B1220"            # deep charcoal-navy canvas
PANEL = "#121B2E"         # slightly lighter panel background
TEXT = "#E7ECF3"
MUTED = "#8B97AC"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid rgba(255,255,255,0.06);
}}

.mono {{
    font-family: 'IBM Plex Mono', monospace;
}}

/* Header block with ECG waveform divider */
.hero {{
    padding: 1.75rem 2rem 1.25rem 2rem;
    background: linear-gradient(180deg, {PANEL} 0%, rgba(18,27,46,0) 100%);
    border-radius: 14px;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(255,255,255,0.06);
}}
.hero h1 {{
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}}
.hero p {{
    color: {MUTED};
    margin-top: 0.35rem;
    font-size: 1rem;
}}
.hero .accent {{ color: {PRIMARY}; }}

/* Metric cards */
.metric-card {{
    background: {PANEL};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
}}
.metric-card .label {{
    color: {MUTED};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.metric-card .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: {TEXT};
    margin-top: 0.2rem;
}}

/* Result banners */
.result-banner {{
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    border: 1px solid;
}}
.result-normal {{
    background: rgba(0,229,160,0.08);
    border-color: rgba(0,229,160,0.35);
}}
.result-risk {{
    background: rgba(255,107,107,0.08);
    border-color: rgba(255,107,107,0.35);
}}
.result-banner h3 {{ margin: 0 0 0.25rem 0; }}
.result-banner p {{ margin: 0; color: {MUTED}; }}

hr {{ border-color: rgba(255,255,255,0.08); }}

footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model artifacts + data
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/heart_disease_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    importances = pd.read_csv("model/feature_importances.csv")
    return model, scaler, feature_columns, importances

@st.cache_data
def load_data():
    raw = pd.read_csv("data/heart_disease_uci_original.csv")
    processed = pd.read_csv("data/heart_disease_processed.csv")
    return raw, processed

model, scaler, feature_columns, importances_df = load_artifacts()
raw_df, processed_df = load_data()

CP_LABELS = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
RESTECG_LABELS = {0: "Normal", 1: "ST-T abnormality", 2: "Left ventricular hypertrophy"}
SLOPE_LABELS = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_LABELS = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🫀 Cardio<span class="accent">Risk</span></h1>
    <p>Clinical decision-support demo — predicts heart disease risk from patient vitals using a Random Forest model trained on the UCI Cleveland Heart Disease dataset.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_data, tab_model = st.tabs(["🔍  Predict", "📊  Dataset", "🧠  Model Insights"])

# ----------------------------------------------------------------------------
# TAB 1 — Predict
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("#### Patient Vitals")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 20, 100, 54)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest pain type", list(CP_LABELS.keys()), format_func=lambda x: CP_LABELS[x])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 246)
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG", list(RESTECG_LABELS.keys()), format_func=lambda x: RESTECG_LABELS[x])
        with c2:
            thalach = st.number_input("Max heart rate achieved", 60, 220, 149)
            exang = st.selectbox("Exercise-induced angina", ["No", "Yes"])
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox("ST segment slope", list(SLOPE_LABELS.keys()), format_func=lambda x: SLOPE_LABELS[x])
            ca = st.selectbox("Major vessels colored (fluoroscopy)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", list(THAL_LABELS.keys()), format_func=lambda x: THAL_LABELS[x])

        predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Risk Assessment")

        if predict_clicked:
            sex_val = 1 if sex == "Male" else 0
            fbs_val = 1 if fbs == "Yes" else 0
            exang_val = 1 if exang == "Yes" else 0

            row = {
                "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs_val, "restecg": restecg, "thalach": thalach, "exang": exang_val,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }
            input_df = pd.DataFrame([row])[feature_columns]
            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0][1]
            prediction = int(proba >= 0.5)

            gauge_color = RISK if prediction == 1 else PRIMARY
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 46, "family": "IBM Plex Mono", "color": TEXT}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": PANEL,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(0,229,160,0.12)"},
                        {"range": [40, 70], "color": "rgba(255,193,7,0.10)"},
                        {"range": [70, 100], "color": "rgba(255,107,107,0.12)"},
                    ],
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=20, r=20, t=30, b=10),
                font={"color": TEXT},
            )
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-banner result-risk">
                    <h3>⚠️ Elevated risk of heart disease</h3>
                    <p>Model confidence: {proba:.1%}. This is a screening signal, not a diagnosis — recommend clinical follow-up.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-banner result-normal">
                    <h3>✅ Low risk of heart disease</h3>
                    <p>Predicted attack probability: {proba:.1%}, based on the vitals entered.</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("What drove this prediction?"):
                top_features = importances_df.head(5)
                fig2 = px.bar(
                    top_features, x="importance", y="feature", orientation="h",
                    color_discrete_sequence=[PRIMARY],
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": TEXT}, height=260, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis={"categoryorder": "total ascending", "title": ""}, xaxis={"title": "Model importance"},
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("These are the model's globally most important features (see the Model Insights tab for the full SHAP analysis), not a per-patient explanation.")
        else:
            st.info("Enter patient vitals on the left and click **Run Prediction** to see the risk assessment.")

# ----------------------------------------------------------------------------
# TAB 2 — Dataset
# ----------------------------------------------------------------------------
with tab_data:
    st.markdown("#### UCI Cleveland Heart Disease Dataset")
    st.write(
        "This app is trained on the classic UCI Cleveland Heart Disease dataset "
        "(303 patients, 14 clinical attributes). The original multi-class target "
        "(0 = no disease, 1–4 = increasing severity) is collapsed to a binary "
        "label (0 = no disease, 1 = disease present) for this model."
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in zip(
        [m1, m2, m3, m4],
        ["Patients", "Features", "With Disease", "Without Disease"],
        [len(processed_df), len(feature_columns), int(processed_df["target"].sum()), int((processed_df["target"] == 0).sum())],
    ):
        col.markdown(f"""<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>""", unsafe_allow_html=True)

    st.markdown("##### Raw dataset (as sourced from UCI)")
    st.dataframe(raw_df, use_container_width=True, height=280)
    st.download_button(
        "⬇ Download original dataset (CSV)",
        raw_df.to_csv(index=False).encode("utf-8"),
        file_name="heart_disease_uci_original.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Target distribution")
        counts = processed_df["target"].map({0: "No Disease", 1: "Disease"}).value_counts()
        fig3 = px.pie(values=counts.values, names=counts.index, hole=0.55,
                      color=counts.index, color_discrete_map={"No Disease": PRIMARY, "Disease": RISK})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT}, height=320,
                            legend={"orientation": "h", "y": -0.1})
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.markdown("##### Age distribution by outcome")
        fig4 = px.histogram(processed_df, x="age", color=processed_df["target"].map({0: "No Disease", 1: "Disease"}),
                             barmode="overlay", color_discrete_map={"No Disease": PRIMARY, "Disease": RISK}, nbins=20)
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": TEXT}, height=320, legend_title="", xaxis_title="Age", yaxis_title="Patients")
        st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Model Insights
# ----------------------------------------------------------------------------
with tab_model:
    st.markdown("#### Model Performance")
    st.caption("Random Forest classifier, evaluated on a held-out 20% stratified test split (random_state=42).")

    m1, m2, m3 = st.columns(3)
    for col, label, value in zip([m1, m2, m3], ["Accuracy", "AUC Score", "Test Set Size"], ["86.7%", "0.941", "60 patients"]):
        col.markdown(f"""<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>""", unsafe_allow_html=True)

    st.markdown("##### Global feature importance")
    fig5 = px.bar(importances_df, x="importance", y="feature", orientation="h", color_discrete_sequence=[PRIMARY])
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": TEXT},
                        height=420, yaxis={"categoryorder": "total ascending", "title": ""}, xaxis={"title": "Importance"})
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("##### SHAP summary")
    st.caption("Each point is one patient in the test set. Red = high feature value, blue = low. Right of center pushes toward disease.")
    st.image("assets/shap_summary.png", use_container_width=True)

    st.markdown("""
    **Reading the model, honestly:**
    - `cp` (chest pain type), `thal`, and `thalach` (max heart rate) are the strongest predictors — consistent with cardiology literature.
    - The dataset is small (297 usable records after cleaning) and sourced from a single hospital cohort, so performance may not generalize to other populations without re-validation.
    - This is a portfolio/demo model, not a certified clinical tool.
    """)

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit · Dataset: UCI Machine Learning Repository — Heart Disease (Cleveland)")
