"""Assemble the feature matrix from cached OHLCV.

Input:  data/raw/sp500_ohlcv_*.parquet (long format from download_data.py)
Output: data/processed/features_<start>_<end>.parquet (long format)

Usage:
    python scripts/build_features.py                       # uses newest raw file
    python scripts/build_features.py --raw data/raw/foo.parquet
    python scripts/build_features.py --horizon 21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlcs.features import add_forward_target, build_feature_panel

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


def pick_latest_raw() -> Path:
    candidates = sorted(RAW_DIR.glob("sp500_ohlcv_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No raw files in {RAW_DIR}. Run scripts/download_data.py first."
        )
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=21, help="forward-return horizon in trading days")
    args = ap.parse_args()

    raw_path = args.raw or pick_latest_raw()
    print(f"Loading {raw_path}")
    df = pd.read_parquet(raw_path)
    df["date"] = pd.to_datetime(df["date"])

    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    print(f"  wide panel: {close.shape[0]} dates · {close.shape[1]} symbols")

    print("Building feature panel...")
    features = build_feature_panel(close, volume)

    print(f"Attaching forward {args.horizon}-day target...")
    features = add_forward_target(features, close, horizon=args.horizon)

    # drop rows where every feature is NaN (pre-warmup) — keep partial-NaN rows
    # so the model can handle missingness itself (LightGBM does)
    feature_cols = [c for c in features.columns if c not in ("date", "symbol")]
    before = len(features)
    features = features.dropna(subset=feature_cols, how="all")
    print(f"  {len(features):,} rows ({before - len(features):,} dropped as all-NaN)")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    stem = raw_path.stem.replace("sp500_ohlcv_", "")
    out = PROC_DIR / f"features_{stem}_h{args.horizon}.parquet"
    features.to_parquet(out, index=False)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
