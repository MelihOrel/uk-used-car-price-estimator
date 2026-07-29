"""UK Used-Car Price Estimator — Streamlit app.

Run locally from the repo root::

    streamlit run app/app.py

The root-level ``app.py`` (Hugging Face Spaces entry point) imports and
runs this same module, so the two never drift apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `streamlit run app/app.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processor import load_config  # noqa: E402
from src.explainability import (build_explainer, explain_prediction,  # noqa: E402
                                plot_prediction_breakdown)
from src.model import PriceModel, load_model  # noqa: E402

st.set_page_config(page_title="UK Used-Car Price Estimator",
                   page_icon="🚗", layout="centered")


# ---------------------------------------------------------------------------
# Cached resources (loaded once per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
def get_resources():
    cfg = load_config(REPO_ROOT / "config.yaml")
    model: PriceModel = load_model(REPO_ROOT / cfg["paths"]["models_dir"])
    with open(REPO_ROOT / cfg["paths"]["model_metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(REPO_ROOT / cfg["paths"]["metrics_file"], encoding="utf-8") as fh:
        metrics = json.load(fh)
    explainer = build_explainer(model)
    return cfg, model, meta, metrics, explainer


try:
    cfg, model, meta, metrics, explainer = get_resources()
except FileNotFoundError:
    st.error("Model artefacts not found. Run `python train.py` first "
             "(this trains the model and writes `models/`).")
    st.stop()

REF_YEAR = cfg["features"]["reference_year"]
lgbm_metrics = metrics["LightGBM (unified)"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚗 UK Used-Car Price Estimator")
st.markdown(
    f"Instant resale-price estimates from a LightGBM model trained on "
    f"**{meta['n_listings']:,} real UK listings** across "
    f"{len(meta['brands'])} brands. Test-set accuracy: "
    f"**MAE £{lgbm_metrics['MAE']:,.0f}**, R² {lgbm_metrics['R2']:.3f}."
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("Your car")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", meta["brands"],
                         format_func=lambda b: b.capitalize())
with col2:
    car_model = st.selectbox("Model", meta["brand_models"][brand])

# sensible defaults for this brand+model (median tax/mpg/engine size)
med = meta["medians_by_brand_model"].get(f"{brand}||{car_model}", {})

col3, col4 = st.columns(2)
with col3:
    yr_min, yr_max = meta["input_ranges"]["year"]
    year = st.number_input("Registration year", min_value=yr_min,
                           max_value=yr_max, value=max(yr_min, yr_max - 4))
with col4:
    mileage = st.number_input("Mileage (miles)", min_value=0,
                              max_value=meta["input_ranges"]["mileage"][1],
                              value=30_000, step=1_000)

col5, col6, col7 = st.columns(3)
with col5:
    transmission = st.selectbox("Transmission", meta["transmissions"])
with col6:
    fuel = st.selectbox("Fuel type", meta["fuel_types"])
with col7:
    es_min, es_max = meta["input_ranges"]["engineSize"]
    engine_size = st.number_input(
        "Engine size (L)", min_value=float(es_min), max_value=float(es_max),
        value=float(med.get("engineSize") or 1.6), step=0.1, format="%.1f")

with st.expander("Optional: road tax & fuel economy"):
    st.caption("Leave the defaults (this model's medians) if unsure — "
               "they matter far less than age, mileage and brand.")
    c1, c2 = st.columns(2)
    with c1:
        tax = st.number_input("Annual road tax (£)", min_value=0.0,
                              max_value=float(meta["input_ranges"]["tax"][1]),
                              value=float(med.get("tax") if med.get("tax")
                                          is not None else 145.0), step=5.0)
    with c2:
        mpg = st.number_input("Fuel economy (mpg)",
                              min_value=float(meta["input_ranges"]["mpg"][0]),
                              max_value=float(meta["input_ranges"]["mpg"][1]),
                              value=float(med.get("mpg") if med.get("mpg")
                                          is not None else 50.0), step=0.5)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Estimate price", type="primary", use_container_width=True):
    X_one = pd.DataFrame([{
        "brand": brand, "model": car_model, "transmission": transmission,
        "fuelType": fuel, "year": int(year), "mileage": float(mileage),
        "tax": float(tax), "mpg": float(mpg),
        "engineSize": float(engine_size), "car_age": REF_YEAR - int(year),
    }])

    iv = model.predict_interval(X_one).iloc[0]
    point, lo, hi = iv["point"], iv["p10"], iv["p90"]

    # graceful out-of-range warnings (model extrapolates, so be honest)
    if mileage > 200_000:
        st.warning("Mileage above ~200k miles is rare in the training data — "
                   "treat this estimate with extra caution.")
    if REF_YEAR - int(year) > 20:
        st.warning("Cars older than ~20 years are sparse in the training "
                   "data — the estimate may be unreliable.")

    st.markdown("---")
    st.metric("Estimated resale price", f"£{point:,.0f}",
              help="Point estimate from the unified LightGBM model")
    st.markdown(
        f"**Likely range: £{lo:,.0f} – £{hi:,.0f}** "
        f"(10th–90th percentile from quantile models; ~8 in 10 comparable "
        f"listings fall inside this band)")

    # per-prediction SHAP explanation
    st.subheader("Why this estimate?")
    breakdown, base = explain_prediction(model, X_one, explainer)
    st.pyplot(plot_prediction_breakdown(breakdown), use_container_width=True)
    st.caption(
        f"Bars show how each detail moved the estimate away from the "
        f"average listing price (£{base:,.0f}). Green pushes the price up, "
        f"red pulls it down.")

# ---------------------------------------------------------------------------
# How it works / limitations
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("How it works & data limitations"):
    st.markdown(f"""
**Pipeline.** Nine brand CSVs (~99k UK listings) are merged with explicit
schema reconciliation (Hyundai's `tax(£)` column, whitespace in model names,
`engineSize == 0` as missing), cleaned, and used to train one unified
LightGBM regressor with `brand` as a feature. Three extra quantile models
(p10/p50/p90) provide the uncertainty band, and SHAP explains each estimate.

**Accuracy.** On a held-out test set of {14657:,} listings:
MAE £{lgbm_metrics['MAE']:,.0f} · RMSE £{lgbm_metrics['RMSE']:,.0f} ·
MAPE {100 * lgbm_metrics['MAPE']:.1f}% · R² {lgbm_metrics['R2']:.3f}.

**Limitations (please read).**
- UK market only; prices in £.
- Listings span roughly 1996–2020 — the model reflects **2020 market
  conditions** and gets no live market updates, so current prices
  (post-2020 inflation, EV shift) will differ.
- Listing price ≠ sale price; condition, service history and colour are
  not in the data.
- Rare combinations (very old cars, exotic engines) get wider, less
  reliable estimates.
""")

st.caption("Built with LightGBM · SHAP · Streamlit — trained on the "
           "100,000 UK Used Car Data Set (Kaggle).")
