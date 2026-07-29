"""Model training and evaluation for the UK Used-Car Price Estimator.

One unified LightGBM regressor is trained on all brands (``brand`` is a
feature) rather than nine per-brand models: low-volume brands (Hyundai has
~4.8k listings vs Ford's ~18k) borrow statistical strength from shared
depreciation patterns, and one model is simpler to version, explain and
deploy.

Uncertainty is provided by three additional LightGBM quantile models
(p10 / p50 / p90) trained with the pinball loss, giving the app a "likely
£low–£high" range per car.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

log = logging.getLogger("model")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """Train / validation / test split, stratified by brand so every brand
    is represented in each partition."""
    s = cfg["split"]
    strat = X["brand"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=s["test_size"], random_state=s["random_state"],
        stratify=strat)
    val_frac_of_tmp = s["val_size"] / (1 - s["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_of_tmp,
        random_state=s["random_state"], stratify=X_tmp["brand"])
    log.info("Split: train=%s val=%s test=%s",
             f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def train_baselines(X_train, y_train, X_test, y_test, cfg: dict) -> dict:
    """Two baselines so the LightGBM lift is measurable:

    1. global-median DummyRegressor (floor);
    2. Ridge regression with one-hot categoricals + scaled numerics
       (a sensible classical benchmark). NaNs are median-imputed because
       linear models can't ingest them (LightGBM can — worth surfacing).
    """
    results = {}

    dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
    results["Median baseline"] = evaluate(y_test, dummy.predict(X_test))

    cats = cfg["features"]["categorical"]
    nums = cfg["features"]["numeric"]
    Xtr, Xte = X_train.copy(), X_test.copy()
    med = Xtr[nums].median()
    Xtr[nums] = Xtr[nums].fillna(med)
    Xte[nums] = Xte[nums].fillna(med)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", StandardScaler(), nums),
    ])
    ridge = make_pipeline(pre, Ridge(alpha=1.0, random_state=42))
    ridge.fit(Xtr, y_train)
    results["Ridge (one-hot)"] = evaluate(y_test, ridge.predict(Xte))
    return results


# ---------------------------------------------------------------------------
# LightGBM point + quantile models
# ---------------------------------------------------------------------------
@dataclass
class PriceModel:
    """Bundles the point model, quantile models and metadata the app needs."""
    point_model: LGBMRegressor
    quantile_models: dict = field(default_factory=dict)  # {0.10: model, ...}
    feature_names: list = field(default_factory=list)
    categorical_features: list = field(default_factory=list)
    category_levels: dict = field(default_factory=dict)  # {col: [levels]}

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align an input frame to training dtypes (needed at app time)."""
        X = X[self.feature_names].copy()
        for col in self.categorical_features:
            X[col] = pd.Categorical(X[col], categories=self.category_levels[col])
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.point_model.predict(self._prep(X))

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._prep(X)
        out = {f"p{int(q * 100)}": m.predict(X)
               for q, m in self.quantile_models.items()}
        out["point"] = self.point_model.predict(X)
        df = pd.DataFrame(out, index=X.index)
        # Quantile models are trained independently; enforce monotone order.
        lo, hi = df["p10"].copy(), df["p90"].copy()
        df["p10"], df["p90"] = np.minimum(lo, hi), np.maximum(lo, hi)
        return df


def _fit_lgbm(params: dict, X_train, y_train, X_val, y_val, cats,
              stopping_rounds: int, **objective_kwargs) -> LGBMRegressor:
    model = LGBMRegressor(**params, **objective_kwargs)
    fit_kwargs = dict(
        eval_metric="l1",
        categorical_feature=cats,
        callbacks=[early_stopping(stopping_rounds, verbose=False),
                   log_evaluation(0)],
    )
    # LightGBM >= 4.7 renames eval_set -> eval_X / eval_y; support both.
    import inspect
    if "eval_X" in inspect.signature(model.fit).parameters:
        model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, **fit_kwargs)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kwargs)
    return model


def train_price_model(X_train, y_train, X_val, y_val, cfg: dict) -> PriceModel:
    params = dict(cfg["model"]["lgbm_params"])
    cats = cfg["features"]["categorical"]
    stop = cfg["model"]["early_stopping_rounds"]

    log.info("Training point model (objective=regression_l1) ...")
    point = _fit_lgbm(params, X_train, y_train, X_val, y_val, cats, stop,
                      objective="regression_l1")
    log.info("Point model stopped at %d trees", point.best_iteration_)

    quantiles = {}
    for q in cfg["model"]["quantiles"]:
        log.info("Training quantile model p%d ...", int(q * 100))
        quantiles[q] = _fit_lgbm(params, X_train, y_train, X_val, y_val, cats,
                                 stop, objective="quantile", alpha=q)

    return PriceModel(
        point_model=point,
        quantile_models=quantiles,
        feature_names=list(X_train.columns),
        categorical_features=cats,
        category_levels={c: list(X_train[c].cat.categories) for c in cats},
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model: PriceModel, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "price_model.joblib"
    joblib.dump(model, path, compress=3)
    log.info("Saved model bundle -> %s", path)
    return path


def load_model(models_dir: Path) -> PriceModel:
    return joblib.load(models_dir / "price_model.joblib")


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Saved metrics -> %s", path)
