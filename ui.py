"""Streamlit UI for interacting with the exoplanet classifier."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import io
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

import preprocess
import train

st.set_page_config(page_title="Exoplanet Classifier", layout="wide")
COSMIC_CSS = """
<style>
body {
    background: radial-gradient(circle at 20% 20%, rgba(12, 19, 43, 0.95), rgba(2, 5, 18, 0.98));
    color: #F8FAFF;
}
section.main > div {
    background: rgba(10, 15, 35, 0.7);
    padding: 1.75rem;
    border-radius: 20px;
    border: 1px solid rgba(127, 219, 255, 0.15);
    box-shadow: 0 0 35px rgba(127, 219, 255, 0.15);
}
.stTabs button {
    background: rgba(18, 28, 60, 0.85) !important;
    color: #E2F2FF !important;
    border-radius: 14px !important;
    border: 1px solid rgba(127, 219, 255, 0.35) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #7FDBFF !important;
}
.stButton button {
    background: linear-gradient(135deg, #7FDBFF 0%, #39CCCC 45%, #B10DC9 100%) !important;
    color: #041021 !important;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    box-shadow: 0 0 20px rgba(127, 219, 255, 0.35);
}
.stMetric {
    background: rgba(18, 28, 60, 0.8);
    border-radius: 14px;
    padding: 0.6rem 1rem;
    border: 1px solid rgba(127, 219, 255, 0.25);
}
.stDataFrame, .stDataFrame div {
    color: #F0F4FF !important;
}
</style>
"""
st.markdown(COSMIC_CSS, unsafe_allow_html=True)
st.title("🌌 Exoplanet Candidate Classification")

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#7FDBFF",
    "#39CCCC",
    "#B10DC9",
    "#FF851B",
    "#FFDC00",
    "#2ECC40",
    "#01FF70",
    "#F012BE",
]
px.defaults.width = None
px.defaults.height = 520


@st.cache_data(show_spinner=False)
def _load_base_dataset() -> pd.DataFrame:
    return preprocess.load_datasets()


@st.cache_data(show_spinner=False)
def _load_star_map_data() -> pd.DataFrame:
    return preprocess.load_star_map_data()


def _show_metrics(metrics: Dict[str, Any], label: str) -> None:
    st.subheader(label)
    cols = st.columns(len(metrics))
    for (name, value), col in zip(metrics.items(), cols):
        col.metric(name.capitalize(), f"{value:.3f}")


base_data = None
try:
    base_data = _load_base_dataset()
except Exception as exc:
    st.warning(f"Unable to load base datasets: {exc}")

star_map_data = None
star_map_error = None
try:
    star_map_data = _load_star_map_data()
except Exception as exc:
    star_map_error = str(exc)

training_tab, prediction_tab, insights_tab = st.tabs(["Training", "Prediction", "Insights"])

with training_tab:
    st.header("Train or Retrain the Model")
    algorithm_label = st.selectbox("Model selection", ("Random Forest (baseline)", "XGBoost"))
    algorithm_key = "random_forest" if "Random Forest" in algorithm_label else "xgboost"
    st.info("Optional: upload an existing model checkpoint (joblib .pkl) to reuse its hyperparameters. XGBoost checkpoints also warm-start training.")
    base_model_file = st.file_uploader("Upload base model (optional)", type=("pkl", "joblib"), key="base_model")
    training_file = st.file_uploader("Upload CSV for retraining (optional)", type="csv")

    if st.button("Train / Retrain", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                base_bytes = base_model_file.getvalue() if base_model_file is not None else None
                if training_file is not None:
                    metrics = train.retrain_model(
                        training_file.getvalue(),
                        algorithm=algorithm_key,
                        base_model=base_bytes,
                    )
                else:
                    df = preprocess.load_datasets()
                    metrics = train.train_model(
                        df,
                        algorithm=algorithm_key,
                        base_model=base_bytes,
                    )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Training failed: {exc}")
            else:
                _show_metrics(metrics, "Latest Training Metrics")
                st.success("Model training complete.")

    # Show previously stored metrics if available
    try:
        past_metrics = train.get_stored_metrics()
    except FileNotFoundError:
        st.info("Train the model to see evaluation metrics here.")
    else:
        _show_metrics(past_metrics, "Stored Metrics")

with prediction_tab:
    st.header("Predict Dispositions")
    st.markdown("Upload a CSV containing the eleven numeric features used for training (`pl_orbper`, `pl_orbsmax`, `pl_rade`, `pl_bmasse`, `pl_eqt`, `pl_insol`, `st_teff`, `st_rad`, `st_mass`, `st_met`, `st_logg`). Extra columns are optional and will be echoed back alongside predictions.")
    prediction_file = st.file_uploader("Upload CSV for prediction", type="csv", key="prediction")

    if st.button("Run Prediction", use_container_width=True):
        if prediction_file is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                payload = train.get_trained_model()
            except FileNotFoundError:
                st.error("Model not trained yet. Train the model before running predictions.")
            else:
                dataframe = pd.read_csv(io.BytesIO(prediction_file.getvalue()))
                features = preprocess.prepare_features_frame(dataframe)
                predictions = payload["model"].predict(features)
                results = dataframe.copy()
                results["prediction"] = predictions
                st.subheader("Predicted Dispositions")
                st.dataframe(results, use_container_width=True)

with insights_tab:
    st.header("Dataset Insights")
    if base_data is None or base_data.empty:
        st.info("Insights will appear once the base datasets are available.")
    else:
        disposition_counts = base_data[preprocess.LABEL_COLUMN].value_counts().reset_index()
        disposition_counts.columns = ["Disposition", "Count"]
        chart = px.bar(
            disposition_counts,
            x="Disposition",
            y="Count",
            title="Distribution of Planet Candidates by Status",
            color="Disposition",
        )
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Orbital Period vs Planet Radius")
        scatter_fig = px.scatter(
            base_data,
            x="pl_orbper",
            y="pl_rade",
            color=preprocess.LABEL_COLUMN,
            title="Verification Dataset View",
            hover_data=["pl_eqt", "pl_insol", "st_teff", "st_rad"],
        )
        scatter_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(scatter_fig, use_container_width=True)

    if star_map_error:
        st.warning(f"Unable to load star map data: {star_map_error}")
    elif star_map_data is not None and not star_map_data.empty:
        st.subheader("Galactic Star Map")
        star_display = star_map_data.copy()
        radius_series = star_display["planet_radius"]
        if radius_series.notna().any():
            median_radius = float(radius_series.median(skipna=True))
        else:
            median_radius = 1.0
        star_display["radius_display"] = radius_series.fillna(median_radius).clip(lower=0.1)
        star_display["discovery_year_label"] = star_display["discovery_year"].apply(
            lambda val: str(int(val)) if pd.notna(val) else "Unknown"
        )
        star_display["identifier"] = star_display["identifier"].fillna("N/A")
        star_fig = px.scatter(
            star_display,
            x="ra",
            y="dec",
            color="discovery_year_label",
            size="radius_display",
            size_max=18,
            hover_name="identifier",
            hover_data={
                "dataset": True,
                preprocess.LABEL_COLUMN: True,
                "planet_radius": True,
                "discovery_year": True,
            },
            labels={
                "ra": "Right Ascension (deg)",
                "dec": "Declination (deg)",
                "discovery_year_label": "Discovery Year",
                "radius_display": "Radius (Earth radii)",
            },
            title="Star Map from Training Catalogues",
        )
        star_fig.update_layout(
            xaxis=dict(range=[360, 0]),
            yaxis_title="Declination (deg)",
            xaxis_title="Right Ascension (deg)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(star_fig, use_container_width=True)
    elif star_map_data is not None:
        st.info("Star map visualisation will appear once RA/Dec data is available.")

    if base_data is not None and not base_data.empty:
        st.subheader("Preview of Combined Dataset")
        st.dataframe(base_data.head(50), use_container_width=True)
