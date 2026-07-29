"""End-to-end training entry point.

Usage (from the repo root)::

    python train.py

Pipeline: merge & clean raw CSVs -> feature engineering + EDA panel ->
baselines -> unified LightGBM (+ p10/p50/p90 quantile models) ->
test-set evaluation -> SHAP global summary -> persist artefacts.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path

import lightgbm
import pandas as pd

from src.data_processor import REPO_ROOT, build_dataset, load_config
from src.explainability import make_global_summary
from src.features import engineer_features, feature_matrix, make_eda_panel
from src.model import (evaluate, save_metrics, save_model, split_data,
                       train_baselines, train_price_model)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train")


def main() -> None:
    t0 = time.time()
    cfg = load_config()

    # 1) data engineering ---------------------------------------------------
    log.info("=" * 70)
    log.info("STEP 1/5 — Merge & clean raw CSVs")
    df = build_dataset(cfg)

    # 2) features + EDA -----------------------------------------------------
    log.info("=" * 70)
    log.info("STEP 2/5 — Feature engineering + EDA")
    df = engineer_features(df, cfg)
    make_eda_panel(df, cfg)
    X, y = feature_matrix(df, cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)

    # 3) baselines ----------------------------------------------------------
    log.info("=" * 70)
    log.info("STEP 3/5 — Baselines")
    metrics = train_baselines(X_train, y_train, X_test, y_test, cfg)
    for name, m in metrics.items():
        log.info("%-18s MAE=£%-8.0f RMSE=£%-8.0f MAPE=%5.1f%%  R²=%.3f",
                 name, m["MAE"], m["RMSE"], 100 * m["MAPE"], m["R2"])

    # 4) unified LightGBM + quantile intervals ------------------------------
    log.info("=" * 70)
    log.info("STEP 4/5 — LightGBM (point + p10/p50/p90)")
    model = train_price_model(X_train, y_train, X_val, y_val, cfg)
    metrics["LightGBM (unified)"] = evaluate(y_test, model.predict(X_test))
    m = metrics["LightGBM (unified)"]
    log.info("%-18s MAE=£%-8.0f RMSE=£%-8.0f MAPE=%5.1f%%  R²=%.3f",
             "LightGBM", m["MAE"], m["RMSE"], 100 * m["MAPE"], m["R2"])

    # interval coverage on the test set (p10–p90 should cover ~80%)
    iv = model.predict_interval(X_test)
    coverage = float(((y_test >= iv["p10"]) & (y_test <= iv["p90"])).mean())
    metrics["interval"] = {"nominal": 0.80, "empirical_coverage": coverage,
                           "mean_width_gbp": float((iv["p90"] - iv["p10"]).mean())}
    log.info("p10–p90 interval: empirical coverage=%.1f%% (nominal 80%%), "
             "mean width=£%.0f", 100 * coverage,
             metrics["interval"]["mean_width_gbp"])

    # 5) explainability + persistence --------------------------------------
    log.info("=" * 70)
    log.info("STEP 5/5 — SHAP + persist artefacts")
    make_global_summary(model, X_test,
                        REPO_ROOT / cfg["paths"]["figures_dir"] / "shap_summary.png")

    models_dir = REPO_ROOT / cfg["paths"]["models_dir"]
    save_model(model, models_dir)
    save_metrics(metrics, REPO_ROOT / cfg["paths"]["metrics_file"])

    # metadata the app uses to build its dropdowns / input ranges
    brand_models = {
        b: sorted(df.loc[df["brand"] == b, "model"].astype(str).unique().tolist())
        for b in sorted(df["brand"].astype(str).unique())
    }
    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lightgbm_version": lightgbm.__version__,
        "python_version": platform.python_version(),
        "n_listings": int(len(df)),
        "brands": sorted(df["brand"].astype(str).unique().tolist()),
        "brand_models": brand_models,
        "input_ranges": {
            "year": [int(df["year"].min()), int(df["year"].max())],
            "mileage": [0, int(df["mileage"].max())],
            "engineSize": [float(df["engineSize"].min()),
                           float(df["engineSize"].max())],
            "tax": [0.0, float(df["tax"].max())],
            "mpg": [float(df["mpg"].min()), float(df["mpg"].max())],
        },
        "transmissions": sorted(df["transmission"].astype(str).unique().tolist()),
        "fuel_types": sorted(df["fuelType"].astype(str).unique().tolist()),
        "medians_by_brand_model": {},  # filled below for tax/mpg defaults
    }
    med = (df.groupby(["brand", "model"], observed=True)[["tax", "mpg", "engineSize"]]
             .median().round(1))
    metadata["medians_by_brand_model"] = {
        f"{b}||{mdl}": {k: (None if pd.isna(v) else float(v))
                        for k, v in row.items()}
        for (b, mdl), row in med.iterrows()
    }
    with open(REPO_ROOT / cfg["paths"]["model_metadata"], "w",
              encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Saved app metadata -> %s", cfg["paths"]["model_metadata"])

    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
