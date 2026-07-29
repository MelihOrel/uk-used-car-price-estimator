"""Data engineering for the UK Used-Car Price Estimator.

Responsibilities
----------------
1. Load nine full-schema brand CSVs and merge them into one dataset,
   adding a ``brand`` column derived from each file's origin.
2. Reconcile schema mismatches explicitly:
   - ``hyundi.csv`` names its road-tax column ``tax(£)`` -> renamed to ``tax``.
   - ``cclass.csv`` / ``focus.csv`` lack ``tax`` and ``mpg`` -> if included,
     the columns are added as NaN together with a ``was_missing_tax_mpg``
     indicator (they are excluded by default because they duplicate rows
     already present in ``ford.csv`` / ``merc.csv``).
3. Clean impossible values (``engineSize == 0`` as missing-encoded-as-zero,
   implausible years / mileages / prices), strip whitespace in string
   columns, drop exact duplicates — logging counts for every action.
4. Provide ``clean_raw_listing`` / ``clean_raw_file``: parse the raw scraped
   ``unclean_*.csv`` format (string prices like ``" £8,000"``, comma-grouped
   numbers, values split across duplicate columns) into the clean schema.
5. Persist the result to ``data/processed/cars_clean.parquet``.

Run as a script from the repo root::

    python -m src.data_processor
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("data_processor")

REPO_ROOT = Path(__file__).resolve().parents[1]

CLEAN_SCHEMA = [
    "brand", "model", "year", "price", "transmission",
    "mileage", "fuelType", "tax", "mpg", "engineSize",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Loading & merging the full-schema brand files
# ---------------------------------------------------------------------------
def load_brand_file(path: Path, brand: str) -> pd.DataFrame:
    """Load one brand CSV, tag it with ``brand`` and reconcile its schema."""
    df = pd.read_csv(path)

    # --- schema reconciliation -------------------------------------------
    if "tax(£)" in df.columns:  # hyundi.csv quirk
        df = df.rename(columns={"tax(£)": "tax"})
        log.info("%s: renamed column 'tax(£)' -> 'tax'", path.name)

    df["brand"] = brand

    missing = [c for c in ("tax", "mpg") if c not in df.columns]
    if missing:  # reduced-schema files (cclass.csv / focus.csv)
        for col in missing:
            df[col] = np.nan
        df["was_missing_tax_mpg"] = True
        log.info("%s: added missing columns %s as NaN (+ indicator)",
                 path.name, missing)
    else:
        df["was_missing_tax_mpg"] = False

    return df[CLEAN_SCHEMA + ["was_missing_tax_mpg"]]


def merge_brand_files(cfg: dict) -> pd.DataFrame:
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    frames = [
        load_brand_file(raw_dir / fname, brand)
        for brand, fname in cfg["data"]["brand_files"].items()
    ]
    if cfg["data"].get("include_reduced_files", False):
        for brand, fname in cfg["data"]["reduced_files"].items():
            frames.append(load_brand_file(raw_dir / fname, brand))
    else:
        log.info("Reduced-schema files (focus.csv, cclass.csv) EXCLUDED: "
                 "they duplicate listings already present in ford.csv / "
                 "merc.csv (set data.include_reduced_files: true to fold in).")
    merged = pd.concat(frames, ignore_index=True)
    log.info("Merged %d files -> %s rows", len(frames), f"{len(merged):,}")
    return merged


# ---------------------------------------------------------------------------
# Cleaning the merged dataset
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply all cleaning rules, logging row/value counts for each."""
    c = cfg["cleaning"]
    df = df.copy()
    n0 = len(df)

    # 1) strip leading/trailing whitespace in string columns (' A1' -> 'A1')
    for col in ("model", "transmission", "fuelType", "brand"):
        stripped = df[col].astype("string").str.strip()
        changed = int((stripped != df[col]).sum())
        df[col] = stripped
        if changed:
            log.info("Stripped whitespace in '%s' for %s values",
                     col, f"{changed:,}")

    # 2) engineSize == 0 is missing-encoded-as-zero -> NaN
    zero_engine = int((df["engineSize"] == 0).sum())
    df.loc[df["engineSize"] == 0, "engineSize"] = np.nan
    log.info("engineSize == 0 (missing encoded as zero): %s values -> NaN",
             f"{zero_engine:,}")

    # 3) implausible values -> drop rows (each rule logged separately)
    rules = {
        f"year outside [{c['year_min']}, {c['year_max']}]":
            ~df["year"].between(c["year_min"], c["year_max"]),
        f"mileage > {c['mileage_max']:,} or negative":
            (df["mileage"] > c["mileage_max"]) | (df["mileage"] < 0),
        f"price outside [£{c['price_min']:,}, £{c['price_max']:,}]":
            ~df["price"].between(c["price_min"], c["price_max"]),
        f"engineSize > {c['engine_size_max']} L":
            df["engineSize"] > c["engine_size_max"],
    }
    bad = pd.Series(False, index=df.index)
    for name, mask in rules.items():
        mask = mask.fillna(False)
        log.info("Implausible: %s -> %s rows flagged", name, f"{int(mask.sum()):,}")
        bad |= mask
    df = df[~bad]
    log.info("Dropped %s implausible rows in total", f"{int(bad.sum()):,}")

    # 4) exact duplicates
    dups = int(df.duplicated(subset=CLEAN_SCHEMA).sum())
    df = df.drop_duplicates(subset=CLEAN_SCHEMA)
    log.info("Dropped %s exact duplicate listings", f"{dups:,}")

    log.info("Cleaning complete: %s -> %s rows", f"{n0:,}", f"{len(df):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raw scraped listings ("unclean_*.csv") -> clean schema
# ---------------------------------------------------------------------------
_MONEY_RE = re.compile(r"[^\d.]")


def _parse_number(value) -> float:
    """Parse ' £8,000', '38,852', '1.0', '' -> float or NaN."""
    if pd.isna(value):
        return np.nan
    s = _MONEY_RE.sub("", str(value))
    return float(s) if s else np.nan


def clean_raw_listing(row: pd.Series, brand: str) -> dict:
    """Turn ONE raw scraped row into the clean schema.

    Raw quirks handled:
    - ``price`` as a string like ``" £8,000"`` (currency symbol + thousands
      separator + leading whitespace).
    - Values split across duplicate columns: ``mileage`` is empty and the
      real value lives in ``mileage2`` as a comma-grouped string; same
      pattern for ``fuel type``/``fuel type2`` and ``engine size``/
      ``engine size2`` — the first non-missing wins.
    - ``engine size`` float artifacts like ``0.999`` -> rounded to 2 dp
      (0.999 -> 1.0).
    - No ``tax``/``mpg`` in the raw feed -> NaN.
    """
    def first_valid(*names):
        for n in names:
            v = row.get(n)
            if pd.notna(v) and str(v).strip() != "":
                return v
        return np.nan

    engine = _parse_number(first_valid("engine size", "engine size2"))
    year = _parse_number(row.get("year"))
    return {
        "brand": brand,
        "model": str(row["model"]).strip(),
        "year": np.nan if pd.isna(year) else int(year),
        "price": _parse_number(row.get("price")),
        "transmission": str(row["transmission"]).strip(),
        "mileage": _parse_number(first_valid("mileage", "mileage2")),
        "fuelType": str(first_valid("fuel type", "fuel type2")).strip(),
        "tax": np.nan,
        "mpg": np.nan,
        "engineSize": np.nan if pd.isna(engine) else round(engine, 2),
    }


def clean_raw_file(path: Path, brand: str) -> pd.DataFrame:
    """Parse a whole ``unclean_*.csv`` file into the clean schema."""
    raw = pd.read_csv(path)
    out = pd.DataFrame([clean_raw_listing(r, brand) for _, r in raw.iterrows()])
    n_bad = int(out[["price", "year"]].isna().any(axis=1).sum())
    out = out.dropna(subset=["price", "year"])
    out["year"] = out["year"].astype(int)
    log.info("%s: parsed %s raw listings -> clean schema "
             "(%s dropped for unparseable price/year)",
             path.name, f"{len(raw):,}", f"{n_bad:,}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_dataset(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()

    merged = merge_brand_files(cfg)
    clean = clean_dataset(merged, cfg)

    # Demonstrate the raw-listing pipeline (not folded into training data —
    # these listings duplicate ford.csv / merc.csv rows).
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    demo_brand = {"unclean_focus.csv": "ford", "unclean_cclass.csv": "mercedes"}
    for fname in cfg["data"]["unclean_files"]:
        demo = clean_raw_file(raw_dir / fname, demo_brand[fname])
        log.info("%s demo output head:\n%s", fname, demo.head(3).to_string())

    out_path = REPO_ROOT / cfg["paths"]["clean_dataset"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    log.info("Saved clean dataset -> %s", out_path)

    print(f"\nFinal shape: {clean.shape}")
    print("\nListings per brand:")
    print(clean["brand"].value_counts().to_string())
    return clean


if __name__ == "__main__":
    build_dataset()
