"""SHAP explainability for the UK Used-Car Price Estimator.

Two deliverables:

1. ``make_global_summary`` — beeswarm summary of what drives price across
   the whole test set, saved to ``reports/figures/shap_summary.png``.
2. ``explain_prediction`` — per-car SHAP breakdown returned as a tidy
   DataFrame (feature, value, shap £-contribution) so the Streamlit app can
   render "low mileage +£1,200, older year −£2,000" style explanations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.model import PriceModel

log = logging.getLogger("explainability")

# Human-readable labels for the app
FEATURE_LABELS = {
    "brand": "Brand",
    "model": "Model",
    "transmission": "Transmission",
    "fuelType": "Fuel type",
    "year": "Registration year",
    "mileage": "Mileage",
    "tax": "Road tax",
    "mpg": "Fuel economy (mpg)",
    "engineSize": "Engine size",
    "car_age": "Car age",
}


def build_explainer(model: PriceModel) -> shap.TreeExplainer:
    return shap.TreeExplainer(model.point_model)


def make_global_summary(model: PriceModel, X_sample: pd.DataFrame,
                        out_path: Path, max_rows: int = 3000) -> Path:
    """Beeswarm SHAP summary on a sample of listings."""
    if len(X_sample) > max_rows:
        X_sample = X_sample.sample(max_rows, random_state=42)
    X_prep = model._prep(X_sample)

    explainer = build_explainer(model)
    shap_values = explainer.shap_values(X_prep)

    # For the beeswarm colour axis, categoricals are shown as codes.
    X_display = X_prep.copy()
    for col in model.categorical_features:
        X_display[col] = X_display[col].cat.codes

    plt.figure()
    shap.summary_plot(shap_values, X_display,
                      feature_names=[FEATURE_LABELS.get(c, c)
                                     for c in X_prep.columns],
                      show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle("What drives UK used-car prices (SHAP)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved SHAP global summary -> %s", out_path)
    return out_path


def explain_prediction(model: PriceModel, X_one: pd.DataFrame,
                       explainer: shap.TreeExplainer | None = None
                       ) -> tuple[pd.DataFrame, float]:
    """SHAP breakdown for a single car.

    Returns
    -------
    (breakdown, base_value)
        ``breakdown`` has columns [feature, label, value, shap_gbp], sorted
        by absolute contribution; ``base_value`` is the model's expected
        price before seeing any features.
    """
    assert len(X_one) == 1, "explain_prediction expects exactly one row"
    explainer = explainer or build_explainer(model)
    X_prep = model._prep(X_one)
    sv = explainer.shap_values(X_prep)[0]

    breakdown = pd.DataFrame({
        "feature": X_prep.columns,
        "label": [FEATURE_LABELS.get(c, c) for c in X_prep.columns],
        "value": [X_one.iloc[0][c] for c in X_prep.columns],
        "shap_gbp": sv,
    })
    breakdown = breakdown.reindex(
        breakdown["shap_gbp"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    return breakdown, float(np.ravel(explainer.expected_value)[0])


def plot_prediction_breakdown(breakdown: pd.DataFrame, top_n: int = 8):
    """Horizontal bar chart of per-feature £ contributions (for the app)."""
    top = breakdown.head(top_n).iloc[::-1]
    labels = [f"{row.label} = {row.value}" for row in top.itertuples()]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in top["shap_gbp"]]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(top) + 1.2))
    ax.barh(labels, top["shap_gbp"], color=colors)
    ax.axvline(0, color="#444", lw=1)
    for y, v in enumerate(top["shap_gbp"]):
        ax.annotate(f"{'+' if v >= 0 else '−'}£{abs(v):,.0f}",
                    (v, y), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right", fontsize=10)
    ax.set_xlabel("Contribution to estimated price (£)")
    ax.margins(x=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
