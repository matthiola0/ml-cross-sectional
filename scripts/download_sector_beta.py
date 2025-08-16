"""Download GICS sector mapping + compute 252d rolling beta vs ^GSPC.

Output: ``data/processed/sector_beta.parquet`` with schema
``(date, symbol, sector, beta_252d)``.

Used by ``notebooks/06_neutralized_backtest.ipynb`` to residualise each
model's raw score against [sector dummies + market beta] before re-running
the quintile long-short backtest.

Sector source: Wikipedia "List of S&P 500 companies" GICS Sector column
(stable, no API key, deterministic snapshot). Symbols not found on
Wikipedia (delisted / re-IPO'd) get ``sector="Unknown"`` so downstream
groupby still works.

Beta source: 252-trading-day rolling OLS slope of daily simple returns
against ^GSPC (downloaded via yfinance). NaN until the first 252 obs
exist. Reuses ``data/raw/sp500_ohlcv_*.parquet`` for symbol returns.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from qtools.data.loaders.us import get_us_prices

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BETA_WINDOW = 252


def load_wiki_sectors() -> pd.DataFrame:
    """Return DataFrame ``(symbol, sector)`` from Wikipedia."""
    headers = {"User-Agent": "ml-cross-sectional/0.0 (research; +https://github.com/)"}
    html = requests.get(WIKI_URL, headers=headers, timeout=30).text
    tables = pd.read_html(io.StringIO(html), header=0)
    df = tables[0][["Symbol", "GICS Sector"]].copy()
    df.columns = ["symbol", "sector"]
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False).str.upper()
    return df


def load_prices() -> pd.DataFrame:
    paths = sorted(RAW_DIR.glob("sp500_ohlcv_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no sp500_ohlcv parquet found under {RAW_DIR}")
    return pd.read_parquet(paths[-1])


def load_market(start: str, end: str) -> pd.Series:
    """SPY as the market proxy (qtools loader; ^GSPC has been flaky on yfinance).

    SPY tracks S&P 500 with TER 9 bps — for 252d rolling beta the difference
    is well below noise. Returns are computed from the auto-adjusted close
    that ``get_us_prices`` already returns.
    """
    df = get_us_prices(["SPY"], start=start, end=end)
    return df.set_index("date")["close"].rename("market_close")


def rolling_beta(symbol_returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
    """Per-symbol time series of 252d rolling beta vs market.

    Both inputs must be aligned on the same date index. Returns a Series
    of betas indexed by date; first ``window-1`` values are NaN.
    """
    cov = symbol_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def main() -> int:
    print("loading prices ...")
    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"])

    start, end = prices["date"].min(), prices["date"].max() + pd.Timedelta(days=1)
    print(f"downloading SPY market proxy {start.date()} -> {end.date()} ...")
    mkt = load_market(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    mkt_ret = mkt.pct_change().rename("market_ret")

    print("computing daily returns ...")
    prices["ret"] = prices.groupby("symbol")["close"].pct_change()

    print(f"computing {BETA_WINDOW}d rolling beta per symbol ...")
    parts = []
    n_symbols = prices["symbol"].nunique()
    for i, (sym, sub) in enumerate(prices.groupby("symbol", sort=False), 1):
        sub = sub.set_index("date")["ret"].sort_index()
        aligned = pd.concat([sub, mkt_ret], axis=1, join="inner").dropna()
        if len(aligned) < BETA_WINDOW:
            continue
        beta = rolling_beta(aligned["ret"], aligned["market_ret"], BETA_WINDOW)
        parts.append(pd.DataFrame({
            "date": beta.index,
            "symbol": sym,
            "beta_252d": beta.values,
        }))
        if i % 100 == 0:
            print(f"  {i}/{n_symbols}")
    beta_df = pd.concat(parts, ignore_index=True).dropna(subset=["beta_252d"])
    print(f"beta rows: {len(beta_df):,}")

    print("loading Wikipedia GICS sectors ...")
    sectors = load_wiki_sectors()
    print(f"  matched sectors for {sectors['symbol'].nunique()} symbols")

    out = beta_df.merge(sectors, on="symbol", how="left")
    out["sector"] = out["sector"].fillna("Unknown")

    out_path = PROC_DIR / "sector_beta.parquet"
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(out):,} rows)")
    print()
    print("sector counts (unique symbols):")
    print(out.drop_duplicates("symbol")["sector"].value_counts().to_string())
    print()
    print("beta sample (last date):")
    last = out["date"].max()
    print(out[out["date"] == last][["symbol", "sector", "beta_252d"]].describe()[["beta_252d"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
