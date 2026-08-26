"""Streamlit dashboard for the synthetic SIF precursor prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_reports.csv"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "sif_tfidf_vectorizer.joblib"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "sif_logistic_regression.joblib"
sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="OIL SIF Precursor Intelligence",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_reports() -> pd.DataFrame:
    """Load the existing synthetic dataset without generating or retraining."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_classifier():
    """Load saved model artifacts; training is intentionally never called here."""
    return joblib.load(VECTORIZER_PATH), joblib.load(CLASSIFIER_PATH)


def sif_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().eq("yes")


def risk_level(sif_density: float, report_count: int) -> str:
    if report_count >= 3 and sif_density >= 60:
        return "HIGH"
    if report_count >= 2 and sif_density >= 30:
        return "MEDIUM"
    return "LOW"


def ranked_summary(reports: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        reports.groupby(column, dropna=False)
        .agg(
            reports=("report_id", "count"),
            sif_count=("sif_potential", lambda values: int(sif_mask(values).sum())),
        )
        .reset_index()
    )
    summary["sif_density"] = (summary["sif_count"] / summary["reports"] * 100).round(1)
    summary["risk_level"] = summary.apply(
        lambda row: risk_level(float(row["sif_density"]), int(row["reports"])), axis=1
    )
    return summary.sort_values(["sif_density", "reports"], ascending=False).reset_index(drop=True)


def render_metric(label: str, value: str | int | float, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


st.markdown(
    """
    <style>
    .main-title { color: #0b3558; margin-bottom: 0; }
    .subtitle { color: #557084; font-size: 1.08rem; margin-top: 0.2rem; }
    .demo-badge { display: inline-block; background: #fff3cd; color: #7a5200;
                  border: 1px solid #e0b84d; border-radius: 999px; padding: 0.35rem 0.8rem;
                  font-weight: 700; letter-spacing: 0.04em; margin: 0.6rem 0 1.2rem; }
    .section-note { color: #637786; font-size: 0.9rem; }
    </style>
    <h1 class="main-title">OIL SIF Precursor Intelligence</h1>
    <p class="subtitle">AI-powered safety report analysis and precursor identification</p>
    <span class="demo-badge">DEMO — SYNTHETIC DATA</span>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists() or not VECTORIZER_PATH.exists() or not CLASSIFIER_PATH.exists():
    st.error("Required dataset or trained model artifacts are missing. Run the data generator and classifier training first.")
    st.stop()

reports = load_reports()
vectorizer, classifier = load_classifier()
sif_values = sif_mask(reports["sif_potential"])
site_summary = ranked_summary(reports, "site")
activity_summary = ranked_summary(reports, "activity")

st.header("Executive Overview")
kpi_columns = st.columns(5)
with kpi_columns[0]:
    render_metric("Total Reports", f"{len(reports):,}")
with kpi_columns[1]:
    render_metric("SIF Potential Reports", f"{int(sif_values.sum()):,}")
with kpi_columns[2]:
    render_metric("SIF Potential %", f"{sif_values.mean() * 100:.1f}%")
with kpi_columns[3]:
    render_metric("High-Risk Sites", int((site_summary["risk_level"] == "HIGH").sum()))
with kpi_columns[4]:
    render_metric("High-Risk Activities", int((activity_summary["risk_level"] == "HIGH").sum()))

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("SIF Risk Distribution")
    risk_distribution = (
        reports["sif_potential"]
        .map({"Yes": "SIF Potential", "No": "Non-SIF"})
        .value_counts()
        .rename_axis("classification")
        .reset_index(name="reports")
    )
    fig = px.pie(
        risk_distribution,
        names="classification",
        values="reports",
        hole=0.58,
        color="classification",
        color_discrete_map={"SIF Potential": "#d95f59", "Non-SIF": "#7c9bb5"},
    )
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Life-Saving Rule Distribution")
    rule_counts = reports["life_saving_rule"].value_counts().rename_axis("rule").reset_index(name="reports")
    fig = px.bar(
        rule_counts.sort_values("reports"),
        x="reports",
        y="rule",
        orientation="h",
        color="reports",
        color_continuous_scale="Blues",
    )
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("High-Risk Sites")
    st.dataframe(
        site_summary.rename(columns={"site": "Site", "reports": "Reports", "sif_count": "SIF Count", "sif_density": "SIF Density %", "risk_level": "Risk Level"})[
            ["Site", "Reports", "SIF Count", "SIF Density %", "Risk Level"]
        ],
        hide_index=True,
        use_container_width=True,
    )
with right:
    st.subheader("High-Risk Activities")
    st.dataframe(
        activity_summary.rename(columns={"activity": "Activity", "reports": "Reports", "sif_count": "SIF Count", "sif_density": "SIF Density %", "risk_level": "Risk Level"})[
            ["Activity", "Reports", "SIF Count", "SIF Density %", "Risk Level"]
        ],
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.subheader("Recurring Precursor Patterns")
st.markdown('<p class="section-note">Recurring combinations are investigation signals and do not prove causality.</p>', unsafe_allow_html=True)
pattern_columns = ["site", "activity", "life_saving_rule", "barrier_failure"]
patterns = (
    reports.groupby(pattern_columns, dropna=False)
    .agg(
        report_count=("report_id", "count"),
        sif_count=("sif_potential", lambda values: int(sif_mask(values).sum())),
    )
    .reset_index()
)
patterns["sif_density"] = (patterns["sif_count"] / patterns["report_count"] * 100).round(1)
patterns["risk_level"] = patterns.apply(
    lambda row: risk_level(float(row["sif_density"]), int(row["report_count"])), axis=1
)
patterns = patterns[patterns["report_count"] >= 2].copy()
patterns["_rank"] = patterns["risk_level"].map({"HIGH": 3, "MEDIUM": 2, "LOW": 1})
patterns = patterns.sort_values(["_rank", "sif_density", "report_count"], ascending=False).drop(columns="_rank").head(15)
st.dataframe(
    patterns.rename(
        columns={
            "site": "Site",
            "activity": "Activity",
            "life_saving_rule": "Life-Saving Rule",
            "barrier_failure": "Barrier Failure",
            "report_count": "Report Count",
            "sif_count": "SIF Count",
            "sif_density": "SIF Density %",
            "risk_level": "Risk Level",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.divider()
st.subheader("Individual Report Analysis")
selected_id = st.selectbox("Select a safety report", reports["report_id"].tolist())
selected = reports.loc[reports["report_id"] == selected_id].iloc[0]
features = vectorizer.transform([str(selected["report_text"])])
prediction = int(classifier.predict(features)[0])
confidence = float(classifier.predict_proba(features)[0][prediction])

try:
    from src.rule_mapper import map_report_to_rule

    rule_result = map_report_to_rule(str(selected["report_text"]))
except Exception:
    rule_result = {"life_saving_rule": "Requires HSE Review", "similarity_score": 0.0}

st.text_area("Original Report", str(selected["report_text"]), height=120, disabled=True)
detail_columns = st.columns(6)
detail_columns[0].metric("SIF Potential", "Yes" if prediction else "No")
detail_columns[1].metric("Confidence", f"{confidence:.1%}")
detail_columns[2].metric("Life-Saving Rule", rule_result["life_saving_rule"])
detail_columns[3].metric("Rule Similarity", f"{rule_result['similarity_score']:.3f}")
detail_columns[4].metric("Activity", str(selected["activity"]))
detail_columns[5].metric("Site", str(selected["site"]))
st.info(f"Barrier Failure: {selected['barrier_failure']}")

st.warning("AI-generated assessments are decision-support signals and must be reviewed by qualified HSE professionals.")
