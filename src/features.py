"""Feature engineering + EDA for the UK Used-Car Price Estimator.

- Derives ``car_age`` from ``year`` (reference year = 2020, the scrape year).
- Casts categoricals (``brand``, ``model``, ``transmission``, ``fuelType``)
  to pandas ``category`` dtype so LightGBM uses its native categorical
  handling — no one-hot explosion, correct split semantics.
- Produces a 4-panel EDA figure saved to ``reports/figures/eda_panel.png``.

Run as a script from the repo root (after ``python -m src.data_processor``)::

    python -m src.features
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_processor import REPO_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("features")


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add engineered features and set dtypes LightGBM expects."""
    df = df.copy()
    ref_year = cfg["features"]["reference_year"]
    df["car_age"] = ref_year - df["year"]

    for col in cfg["features"]["categorical"]:
        df[col] = df[col].astype("category")

    log.info("Engineered features: car_age = %d - year; categoricals -> "
             "category dtype: %s", ref_year, cfg["features"]["categorical"])
    return df


def feature_matrix(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into (X, y) using the config feature lists."""
    feats = cfg["features"]["categorical"] + cfg["features"]["numeric"]
    return df[feats], df[cfg["features"]["target"]]


# ---------------------------------------------------------------------------
# EDA panel
# ---------------------------------------------------------------------------
def make_eda_panel(df: pd.DataFrame, cfg: dict) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    gbp = mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}")

    # 1) price distribution
    ax = axes[0, 0]
    sns.histplot(df["price"], bins=80, ax=ax, color="#2a6f97")
    ax.set_title("Price distribution")
    ax.set_xlabel("Price")
    ax.xaxis.set_major_formatter(gbp)

    # 2) price vs car age (median + IQR band)
    ax = axes[0, 1]
    grp = df.groupby("car_age")["price"].quantile([0.25, 0.5, 0.75]).unstack()
    grp = grp[grp.index <= 20]
    ax.plot(grp.index, grp[0.5], color="#c1121f", lw=3, label="Median")
    ax.fill_between(grp.index, grp[0.25], grp[0.75], alpha=0.25,
                    color="#c1121f", label="IQR")
    ax.set_title("Price vs car age")
    ax.set_xlabel("Car age (years)")
    ax.set_ylabel("Price")
    ax.yaxis.set_major_formatter(gbp)
    ax.legend()

    # 3) price vs mileage (hexbin — 95k scatter points would saturate)
    ax = axes[1, 0]
    hb = ax.hexbin(df["mileage"], df["price"], gridsize=60, bins="log",
                   cmap="viridis")
    ax.set_title("Price vs mileage (log-count hexbin)")
    ax.set_xlabel("Mileage (miles)")
    ax.set_ylabel("Price")
    ax.yaxis.set_major_formatter(gbp)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.colorbar(hb, ax=ax, label="log10(listings)")

    # 4) price by brand
    ax = axes[1, 1]
    order = df.groupby("brand", observed=True)["price"].median()\
              .sort_values(ascending=False).index
    sns.boxplot(data=df, x="brand", y="price", order=order, ax=ax,
                showfliers=False, color="#588157")
    ax.set_title("Price by brand (fliers hidden)")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    ax.yaxis.set_major_formatter(gbp)
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle("UK Used-Car Listings — EDA", fontsize=24, y=1.0)
    fig.tight_layout()

    out = REPO_ROOT / cfg["paths"]["figures_dir"] / "eda_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved EDA panel -> %s", out)
    return out


if __name__ == "__main__":
    cfg = load_config()
    data = pd.read_parquet(REPO_ROOT / cfg["paths"]["clean_dataset"])
    data = engineer_features(data, cfg)
    make_eda_panel(data, cfg)
