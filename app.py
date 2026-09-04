"""Professional HSE decision-support dashboard for SIH26165."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (  # noqa: E402
    calculate_kpis,
    monthly_trend,
    prepare_reports,
    precursor_summary,
    rank_dimension,
    rule_summary,
)
from src.report_features import extract_report_features  # noqa: E402
from src.review_store import load_reviews, save_review  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "synthetic_reports.csv"
REVIEW_PATH = PROJECT_ROOT / "data" / "review_feedback.csv"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "sif_tfidf_vectorizer.joblib"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "sif_logistic_regression.joblib"

st.set_page_config(page_title="OIL SIF Precursor Intelligence", page_icon="⛑️", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return prepare_reports(pd.read_csv(DATA_PATH))


@st.cache_resource
def load_classifier():
    """Load saved artifacts only; no fitting or retraining happens in the app."""
    return joblib.load(VECTORIZER_PATH), joblib.load(CLASSIFIER_PATH)


@st.cache_data
def load_review_data() -> pd.DataFrame:
    return load_reviews(REVIEW_PATH)


def risk_level(sif_density: float, report_count: int) -> str:
    if report_count >= 3 and sif_density >= 60:
        return "HIGH"
    if report_count >= 2 and sif_density >= 30:
        return "MEDIUM"
    return "LOW"


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.copy()
    query = st.sidebar.text_input("Search reports", placeholder="Text, ID, site, activity...")
    if query:
        matches = filtered.astype(str).apply(lambda column: column.str.contains(query, case=False, na=False))
        filtered = filtered[matches.any(axis=1)]

    sif_filter = st.sidebar.selectbox("SIF Potential", ["All", "Yes", "No"])
    if sif_filter != "All":
        filtered = filtered[filtered["sif_potential"].eq(sif_filter)]

    def select_values(label: str, column: str) -> None:
        nonlocal filtered
        options = sorted(filtered[column].astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, options)
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]

    select_values("Life-Saving Rule", "life_saving_rule")
    select_values("Precursor", "precursor")
    select_values("Activity", "activity")
    select_values("Location", "location")
    select_values("Priority", "priority")

    dates = filtered["date"].dropna()
    if not dates.empty:
        chosen = st.sidebar.date_input("Date range", value=(dates.min().date(), dates.max().date()))
        if isinstance(chosen, tuple) and len(chosen) == 2:
            start, end = pd.Timestamp(chosen[0]), pd.Timestamp(chosen[1]) + pd.Timedelta(days=1)
            filtered = filtered[filtered["date"].between(start, end, inclusive="left")]
    return filtered


def empty_state(message: str) -> None:
    st.info(message)


def report_table(data: pd.DataFrame, limit: int | None = None) -> None:
    if data.empty:
        empty_state("No reports match the current filters.")
        return
    columns = ["report_id", "date", "site", "location", "activity", "sif_potential", "priority", "precursor"]
    display = data[[column for column in columns if column in data.columns]].copy()
    display = display.rename(columns={
        "report_id": "Report ID", "date": "Date", "site": "Site", "location": "Location",
        "activity": "Activity", "sif_potential": "SIF Potential", "priority": "Priority", "precursor": "Precursor",
    })
    st.dataframe(display.head(limit) if limit else display, hide_index=True, use_container_width=True)


def dashboard_page(data: pd.DataFrame) -> None:
    st.header("HSE Intelligence Dashboard")
    st.caption("AI-assisted HSE safety intelligence • Decision support — human verification required.")
    if data.empty:
        empty_state("No reports match the current filters. Clear or broaden the filters to view dashboard analytics.")
        return
    kpis = calculate_kpis(data)
    cards = st.columns(6)
    values = [
        ("Total Reports Analyzed", f"{kpis['total_reports']:,}"),
        ("SIF-Potential Reports", f"{kpis['sif_reports']:,}"),
        ("Non-SIF Reports", f"{kpis['non_sif_reports']:,}"),
        ("High-Priority Reports", f"{kpis['high_priority_reports']:,}"),
        ("Top SIF Precursor", kpis["top_precursor"]),
        ("Top Life-Saving Rule", kpis["top_rule"]),
    ]
    for card, (label, value) in zip(cards, values):
        with card:
            st.metric(label, value)

    st.divider()
    trend = monthly_trend(data)
    if trend.empty:
        empty_state("No valid dates are available for a trend.")
    else:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=trend["month"], y=trend["reports"], mode="lines+markers", name="All reports", line=dict(color="#6f8fa8")))
        figure.add_trace(go.Scatter(x=trend["month"], y=trend["sif_count"], mode="lines+markers", name="SIF potential", line=dict(color="#d95f59", width=3)))
        figure.update_layout(title="SIF-Potential Reports Over Time", xaxis_title="Month", yaxis_title="Reports", hovermode="x unified")
        st.plotly_chart(figure, use_container_width=True)

    first, second = st.columns(2)
    with first:
        rules = rule_summary(data)
        figure = px.bar(rules.sort_values("frequency"), x="frequency", y="life_saving_rule", orientation="h", color="sif_count", color_continuous_scale="Blues", title="Life-Saving Rule Distribution")
        figure.update_layout(yaxis_title="", coloraxis_colorbar_title="SIF count")
        st.plotly_chart(figure, use_container_width=True)
    with second:
        precursors = precursor_summary(data)
        figure = px.bar(precursors.sort_values("frequency"), x="frequency", y="precursor", orientation="h", color="sif_density", color_continuous_scale="Reds", title="Precursor Distribution")
        figure.update_layout(yaxis_title="", coloraxis_colorbar_title="SIF density %")
        st.plotly_chart(figure, use_container_width=True)

    third, fourth = st.columns(2)
    with third:
        sites = rank_dimension(data, "site")
        figure = px.bar(sites.sort_values("sif_density"), x="sif_density", y="site", orientation="h", color="sif_density", color_continuous_scale="Oranges", title="Site / Location Precursor Density")
        figure.update_layout(xaxis_title="Reported SIF density (%)", yaxis_title="")
        st.plotly_chart(figure, use_container_width=True)
    with fourth:
        activities = rank_dimension(data, "activity")
        figure = px.bar(activities.head(12).sort_values("sif_density"), x="sif_density", y="activity", orientation="h", color="sif_density", color_continuous_scale="Oranges", title="Activity Precursor Density")
        figure.update_layout(xaxis_title="Reported SIF density (%)", yaxis_title="")
        st.plotly_chart(figure, use_container_width=True)
    st.caption("Demo/synthetic statistics only. Precursor density is reported precursor concentration, not a validated injury or fatality probability.")


def reports_page(data: pd.DataFrame, classifier_parts) -> None:
    st.header("Report Investigation")
    if data.empty:
        empty_state("No reports match the current filters.")
        return
    report_id = st.selectbox("Select a report", data["report_id"].astype(str).tolist())
    row = data.loc[data["report_id"].astype(str).eq(report_id)].iloc[0]
    vectorizer, classifier = classifier_parts
    features = vectorizer.transform([str(row["report_text"])])
    prediction = int(classifier.predict(features)[0])
    confidence = float(classifier.predict_proba(features)[0][prediction])
    extracted = extract_report_features(row)
    try:
        from src.rule_mapper import map_report_to_rule
        rule_result = map_report_to_rule(str(row["report_text"]))
    except Exception as error:
        rule_result = {"life_saving_rule": "Requires HSE Review", "similarity_score": 0.0}
        st.caption(f"Semantic rule model unavailable for this session: {error}")

    st.text_area("Original Report Text", str(row["report_text"]), height=130, disabled=True)
    cards = st.columns(5)
    cards[0].metric("SIF Potential", "Yes" if prediction else "No")
    cards[1].metric("Confidence", f"{confidence:.1%}")
    cards[2].metric("Priority", str(row["priority"]))
    cards[3].metric("Mapped Life-Saving Rule", rule_result["life_saving_rule"])
    cards[4].metric("Rule Similarity", f"{rule_result['similarity_score']:.3f}")

    st.subheader("Extracted Information")
    info = st.columns(5)
    info[0].metric("Activity", extracted["activity"])
    info[1].metric("Location", extracted["location"])
    info[2].metric("Hazard", extracted["hazard"])
    info[3].metric("Precursor", extracted["precursor"])
    info[4].metric("Barrier Failure", extracted["barrier_failure"])

    st.subheader("Why Was This Flagged?")
    st.caption("Evidence-based signals from the report and model output; this is not a claimed feature-level explanation.")
    evidence_col, explanation_col = st.columns(2)
    with evidence_col:
        st.markdown("**Evidence from report**")
        for sentence in extracted["evidence"]:
            st.markdown(f"> {sentence}")
        st.markdown("**Detected signals**")
        if extracted["signals"]:
            for signal in extracted["signals"]:
                st.markdown(f"- {signal}")
        else:
            st.markdown("- No supported signal detected")
    with explanation_col:
        st.markdown("**Classification**")
        st.write("SIF Potential" if prediction else "Non-SIF")
        st.markdown("**Mapped Rule**")
        st.write(rule_result["life_saving_rule"])
        st.markdown("**Reviewer should verify**")
        st.write("The reported exposure, barrier failure, and priority before taking HSE action.")


def precursor_page(data: pd.DataFrame) -> None:
    st.header("Precursor Pattern Explorer")
    st.caption("Recurring patterns require HSE investigation; they do not prove causality.")
    summary = precursor_summary(data)
    if summary.empty:
        empty_state("No precursor patterns are available for the current filters.")
        return
    st.dataframe(summary.rename(columns={"precursor": "Precursor", "frequency": "Frequency", "sif_count": "SIF Count", "sif_density": "SIF Density %", "activities": "Associated Activities", "locations": "Associated Locations"}), hide_index=True, use_container_width=True)
    selected = st.selectbox("Inspect precursor", summary["precursor"].tolist())
    selected_reports = data[data["precursor"].eq(selected)]
    st.markdown(f"**Supporting reports: {len(selected_reports)}**")
    report_table(selected_reports)
    trend = monthly_trend(selected_reports)
    if len(trend) > 1:
        st.plotly_chart(px.line(trend, x="month", y="reports", markers=True, title=f"Reported frequency over time: {selected}"), use_container_width=True)


def rules_page(data: pd.DataFrame) -> None:
    st.header("Life-Saving Rule Analytics")
    summary = rule_summary(data)
    if summary.empty:
        empty_state("No rule data is available for the current filters.")
        return
    st.dataframe(summary.rename(columns={"life_saving_rule": "Life-Saving Rule", "frequency": "Frequency", "sif_count": "SIF-Potential Count", "percentage": "Distribution %"}), hide_index=True, use_container_width=True)
    selected = st.selectbox("Inspect Life-Saving Rule", summary["life_saving_rule"].tolist())
    selected_reports = data[data["life_saving_rule"].eq(selected)]
    st.markdown(f"**Associated reports: {len(selected_reports)}**")
    report_table(selected_reports)
    trend = monthly_trend(selected_reports)
    if len(trend) > 1:
        st.plotly_chart(px.line(trend, x="month", y=["reports", "sif_count"], markers=True, title=f"Rule trend: {selected}"), use_container_width=True)


def locations_activities_page(data: pd.DataFrame) -> None:
    st.header("Locations & Activities")
    st.caption("Density = SIF-potential reports divided by total reports in the selected group. This is a priority indicator, not an accident probability.")
    locations, activities = st.tabs(["Locations", "Activities"])
    with locations:
        summary = rank_dimension(data, "location")
        st.dataframe(summary.rename(columns={"location": "Location", "reports": "Reports", "sif_count": "SIF Count", "sif_density": "Precursor Density %"}), hide_index=True, use_container_width=True)
        if not summary.empty:
            selected = st.selectbox("Inspect location", summary["location"].tolist())
            report_table(data[data["location"].eq(selected)])
    with activities:
        summary = rank_dimension(data, "activity")
        st.dataframe(summary.rename(columns={"activity": "Activity", "reports": "Reports", "sif_count": "SIF Count", "sif_density": "Precursor Density %"}), hide_index=True, use_container_width=True)
        if not summary.empty:
            selected = st.selectbox("Inspect activity", summary["activity"].tolist())
            report_table(data[data["activity"].eq(selected)])


def trends_page(data: pd.DataFrame) -> None:
    st.header("Trend / Early Warning")
    st.caption("Use trends to identify emerging or repeated reported patterns requiring HSE attention. Trends do not predict accidents.")
    trend = monthly_trend(data)
    if trend.empty:
        empty_state("No valid dates are available for trend analysis.")
        return
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=trend["month"], y=trend["sif_count"], mode="lines+markers", name="SIF potential"))
    chart.add_trace(go.Scatter(x=trend["month"], y=trend["reports"], mode="lines+markers", name="All reports"))
    chart.update_layout(title="Reported SIF-Potential Trend", xaxis_title="Month", yaxis_title="Reports")
    st.plotly_chart(chart, use_container_width=True)
    rule = st.selectbox("Optional Life-Saving Rule trend", ["All"] + sorted(data["life_saving_rule"].unique().tolist()))
    if rule != "All":
        selected_trend = monthly_trend(data, "life_saving_rule", rule)
        st.plotly_chart(px.line(selected_trend, x="month", y=["reports", "sif_count"], markers=True, title=f"Reported trend: {rule}"), use_container_width=True)


def review_page(data: pd.DataFrame, classifier_parts) -> None:
    st.header("HSE Human-in-the-Loop Review")
    st.caption("Reviewer decisions are stored separately and never overwrite the original AI assessment.")
    reviews = load_review_data()
    reviewed_ids = set(reviews["report_id"].astype(str)) if not reviews.empty else set()
    pending = data[~data["report_id"].astype(str).isin(reviewed_ids)]
    st.metric("Reports awaiting review", len(pending))
    if pending.empty:
        empty_state("No pending reports match the current filters.")
        return
    report_id = st.selectbox("Select report for review", pending["report_id"].astype(str).tolist())
    row = pending.loc[pending["report_id"].astype(str).eq(report_id)].iloc[0]
    vectorizer, classifier = classifier_parts
    features = vectorizer.transform([str(row["report_text"])])
    prediction = int(classifier.predict(features)[0])
    confidence = float(classifier.predict_proba(features)[0][prediction])
    st.info(f"AI Prediction: {'SIF Potential' if prediction else 'Non-SIF'} • Confidence: {confidence:.1%}")
    st.write(str(row["report_text"]))
    with st.form("review_form"):
        reviewer = st.text_input("Reviewer Name")
        decision = st.radio("Decision", ["Confirm AI Assessment", "Correct Assessment"])
        corrected = st.selectbox("Corrected Classification", ["SIF Potential", "Non-SIF"])
        comment = st.text_area("Reviewer Comment")
        submitted = st.form_submit_button("Save Review")
    if submitted:
        if not reviewer.strip() or not comment.strip():
            st.error("Reviewer name and comment are required.")
        else:
            final_classification = corrected if decision == "Correct Assessment" else ("SIF Potential" if prediction else "Non-SIF")
            save_review(REVIEW_PATH, {
                "report_id": report_id,
                "ai_prediction": "SIF Potential" if prediction else "Non-SIF",
                "ai_confidence": round(confidence, 4),
                "reviewer_name": reviewer.strip(),
                "reviewer_decision": decision,
                "corrected_classification": final_classification,
                "reviewer_comment": comment.strip(),
            })
            load_review_data.clear()
            st.success("Review saved. The original AI output remains unchanged.")


def about_page() -> None:
    st.header("About")
    st.markdown("**OIL SIF Precursor Intelligence** is an AI-assisted HSE safety intelligence prototype for decision support.")
    st.markdown("**Current data status:** DEMO / SYNTHETIC DATA — not official OIL data.")
    st.markdown("**Pipeline:** Safety report → TF-IDF + Logistic Regression SIF classifier → semantic Life-Saving Rule mapping → structured precursor analysis → HSE review.")
    st.markdown("**Human verification:** AI-generated assessments are decision-support signals and must be reviewed by qualified HSE professionals.")
    st.markdown("The system identifies potential SIF precursors and recurring patterns to help prioritize investigation. It does not predict accidents, guarantee compliance, or replace HSE professionals.")


st.markdown("<style>.main-title{color:#0b3558;margin-bottom:0}.subtitle{color:#557084;font-size:1.1rem}.demo-badge{display:inline-block;background:#fff3cd;color:#7a5200;border:1px solid #e0b84d;border-radius:999px;padding:.35rem .8rem;font-weight:700;letter-spacing:.04em;margin:.6rem 0 1rem}</style>", unsafe_allow_html=True)
st.markdown('<h1 class="main-title">OIL SIF Precursor Intelligence</h1><p class="subtitle">AI-assisted HSE safety intelligence</p><span class="demo-badge">PROTOTYPE / DEMO DATA</span>', unsafe_allow_html=True)

if not DATA_PATH.exists() or not VECTORIZER_PATH.exists() or not CLASSIFIER_PATH.exists():
    st.error("Required dataset or trained model artifacts are missing. Run the generator and classifier training first.")
    st.stop()

try:
    all_reports = load_data()
    reports = apply_filters(all_reports)
    classifier_parts = load_classifier()
except Exception as error:
    st.error(f"The dashboard could not load its data or model artifacts: {error}")
    st.stop()

st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Dashboard", "Reports", "Precursor Explorer", "Life-Saving Rules", "Locations & Activities", "Trend / Early Warning", "Review Queue", "About"])
st.sidebar.caption(f"Showing {len(reports):,} of {len(all_reports):,} reports")
st.sidebar.caption("Decision support — human verification required.")

if page == "Dashboard":
    dashboard_page(reports)
elif page == "Reports":
    reports_page(reports, classifier_parts)
elif page == "Precursor Explorer":
    precursor_page(reports)
elif page == "Life-Saving Rules":
    rules_page(reports)
elif page == "Locations & Activities":
    locations_activities_page(reports)
elif page == "Trend / Early Warning":
    trends_page(reports)
elif page == "Review Queue":
    review_page(reports, classifier_parts)
else:
    about_page()
