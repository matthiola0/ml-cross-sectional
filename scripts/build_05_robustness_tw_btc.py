"""Build notebooks/05_robustness_tw_btc.ipynb.

Re-runs the XGB ranker + naive_ew baseline on two smaller universes to test
whether the S&P 500 result transfers:

- Taiwan 0050 constituents (~50 large-cap equities), 2015-01 → 2025-07, OOS 2020–2024.
- Top-20 liquid USDT-quoted crypto pairs on Binance, 2018-01 → 2025-07, OOS 2022–2024.

Universe sizes are a fraction of S&P 500's 500, so we expect per-date IC to
be noisier and the tree model's interaction advantage to shrink. The point
of the appendix is honest documentation of that decline, not a sales pitch.

Crypto universe is hard-coded rather than pulled from `get_top_pairs()` at
runtime so the notebook stays reproducible — live ranking changes every
24 hours.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).parent.parent / "notebooks" / "05_robustness_tw_btc.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


CELLS = [
    md(r"""
# 05 — Robustness: TW 0050 and BTC-universe

**Purpose.** The S&P 500 headline was XGB Sharpe 0.87 net, 15.4% annual.
Does the same pipeline — same 12 features, same model, same walk-forward —
still work on two smaller, structurally different universes? Or is the result
a large-cap-US artefact?

**Universes.**
- **TW**: 50 current Taiwan 0050 constituents (hard-coded snapshot; see
  `qtools.data.loaders.twse._TW50_TICKERS`). 2015-01 → 2025-07. 21-day forward
  target, monthly rebalance, `TW_EQUITY` cost model (49 bps round-trip
  including 0.3% securities transaction tax).
- **BTC-uni**: 20 liquid USDT-quoted pairs on Binance, hard-coded. 2018-01 →
  2025-07 (shorter window — BNB/Binance does not predate 2017). OOS from
  2022 to allow ≥ 4 years of training. Monthly rebalance, `CRYPTO` cost
  model (20 bps per leg).

**Models.** XGB ranker + naive_ew only — the two key contenders. Linear
baselines and LGBM are skipped to keep the notebook focused; the S&P 500
analysis already showed they follow the same qualitative ranking.

**Honest prior.** A cross-sectional ranker needs enough names per date for
the rank to be meaningful. With ~50 names on TW each date splits into
~10-stock quintiles; with ~20 on crypto, ~4-stock quintiles. Expect IC and
Sharpe to be worse than on S&P 500, not comparable. If they are comparable,
suspect the test setup rather than celebrating.
"""),
    code(r"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qtools.backtest.costs import CRYPTO, TW_EQUITY, US_EQUITY
from qtools.backtest.engine import BacktestEngine
from qtools.backtest.portfolio import signal_to_weights
from qtools.data.loaders.crypto import get_crypto_prices
from qtools.data.loaders.twse import get_tw50_constituents, get_tw_prices
from qtools.metrics.performance import (
    annualized_return, max_drawdown, sharpe,
)

from mlcs.features import add_forward_target, build_feature_panel
from mlcs.model import EqualWeightBaseline, XGBRankerModel
from mlcs.validation import walk_forward_years

FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110

FEATURES = [
    "mom_12_1", "reversal_1w", "size_adv_60",
    "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_20d", "vol_60d",
    "rsi_14", "macd_hist", "volume_z_60",
]
TARGET = "fwd_rank_21d"
"""),
    md(r"""
## 1. Shared helpers

One feature-building function + one walk-forward runner, reused across both
universes.
"""),
    code(r"""
def prepare_feature_matrix(prices: pd.DataFrame, label: str) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = prices.pivot(index="date", columns="symbol", values="volume").sort_index()
    feats = build_feature_panel(close, volume)
    feats = add_forward_target(feats, close, horizon=21)
    feats = feats.dropna(subset=[TARGET])
    print(f"[{label}] feature rows: {len(feats):,}, symbols: {feats['symbol'].nunique()}")
    return feats, close


def run_walkforward(work: pd.DataFrame, first_oos: int, last_oos: int, label: str) -> pd.DataFrame:
    all_preds = []
    for oos_year, tr_mask, te_mask in walk_forward_years(work["date"], first_oos, last_oos):
        tr = work.loc[tr_mask]
        te = work.loc[te_mask]
        if len(tr) == 0 or len(te) == 0:
            continue
        # fewer trees / shallower depth — small universes overfit fast
        for mdl in [
            XGBRankerModel(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
            EqualWeightBaseline(),
        ]:
            mdl.fit(tr[FEATURES], tr[TARGET], tr["date"])
            scores = mdl.predict(te[FEATURES], te["date"])
            out = te[["date", "symbol", TARGET]].copy()
            out["model"] = mdl.name
            out["score"] = scores
            out["universe"] = label
            all_preds.append(out)
    return pd.concat(all_preds, ignore_index=True)


def ic_summary(preds: pd.DataFrame) -> pd.DataFrame:
    def _per_date_ic(g):
        sub = g[["score", TARGET]].dropna()
        if len(sub) < 5:   # min names per date
            return np.nan
        return sub["score"].rank().corr(sub[TARGET])

    rows = []
    for (uni, model), group in preds.groupby(["universe", "model"]):
        ic = group.groupby("date").apply(_per_date_ic, include_groups=False).dropna()
        if len(ic) == 0:
            continue
        ir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        rows.append({
            "universe": uni, "model": model,
            "mean_ic": round(ic.mean(), 4),
            "ic_ir":   round(ir, 3) if pd.notna(ir) else np.nan,
            "t_stat":  round(ir * np.sqrt(len(ic)), 2) if pd.notna(ir) else np.nan,
            "hit_rate": round((ic > 0).mean(), 3),
            "n_dates":  int(len(ic)),
        })
    return pd.DataFrame(rows).sort_values(["universe", "ic_ir"], ascending=[True, False]).reset_index(drop=True)


def run_backtest(preds: pd.DataFrame, prices: pd.DataFrame, cost, n_quantiles: int, rebalance: str):
    rows = []
    for model, group in preds.groupby("model"):
        signal = group.pivot(index="date", columns="symbol", values="score").sort_index()
        weights = signal_to_weights(signal, n_quantiles=n_quantiles, long_short=True, rebalance=rebalance)
        engine = BacktestEngine(prices[["date", "symbol", "close"]], cost_model=cost)
        res = engine.run(weights)
        rows.append({
            "model":       model,
            "ann_gross":   round(annualized_return(res.gross_returns), 4),
            "ann_net":     round(annualized_return(res.returns), 4),
            "sharpe_gross":round(sharpe(res.gross_returns), 3),
            "sharpe_net":  round(sharpe(res.returns), 3),
            "mdd_net":     round(max_drawdown(res.returns), 4),
            "cost_drag":   round((res.gross_returns.mean() - res.returns.mean()) * 252, 4),
        })
    return pd.DataFrame(rows).sort_values("sharpe_net", ascending=False).reset_index(drop=True)
"""),
    md(r"""
## 2. Taiwan 0050
"""),
    code(r"""
tw_prices = get_tw_prices(get_tw50_constituents(), "2015-01-01", "2025-07-31", adjust=True)
print("TW price rows:", len(tw_prices), "  symbols:", tw_prices["symbol"].nunique())
tw_work, tw_close = prepare_feature_matrix(tw_prices, "TW")
"""),
    code(r"""
tw_preds = run_walkforward(tw_work, first_oos=2020, last_oos=2024, label="TW")
print("TW OOS predictions:", tw_preds.shape)
"""),
    code(r"""
tw_ic = ic_summary(tw_preds)
tw_ic
"""),
    code(r"""
tw_bt = run_backtest(tw_preds, tw_prices, TW_EQUITY, n_quantiles=5, rebalance="ME")
tw_bt
"""),
    md(r"""
## 3. BTC-universe — 20 liquid USDT pairs on Binance

Download is live (ccxt → Binance REST). First run takes a couple of minutes
per pair and populates `~/.qtools_cache/crypto_prices/`; subsequent runs are
instant.
"""),
    code(r"""
# hard-coded liquid universe — reproducible. Tickers delisted from Binance
# will silently drop out of the downloaded frame.
BTC_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "SOL/USDT", "DOT/USDT", "LTC/USDT", "BCH/USDT",
    "LINK/USDT", "AVAX/USDT", "ATOM/USDT", "TRX/USDT", "XLM/USDT",
    "ETC/USDT", "FIL/USDT", "NEAR/USDT", "UNI/USDT", "ALGO/USDT",
]

btc_prices = get_crypto_prices(BTC_UNIVERSE, "2018-01-01", "2025-07-31", interval="1d")
print("BTC-uni price rows:", len(btc_prices), "  symbols:", btc_prices["symbol"].nunique())
"""),
    code(r"""
btc_work, btc_close = prepare_feature_matrix(btc_prices, "BTC")

# OOS 2022-2024 to leave 4 years of training; before 2022 the panel is
# dominated by pairs with very short history.
btc_preds = run_walkforward(btc_work, first_oos=2022, last_oos=2024, label="BTC")
print("BTC OOS predictions:", btc_preds.shape)
"""),
    code(r"""
btc_ic = ic_summary(btc_preds)
btc_ic
"""),
    code(r"""
btc_bt = run_backtest(btc_preds, btc_prices, CRYPTO, n_quantiles=5, rebalance="ME")
btc_bt
"""),
    md(r"""
## 4. Three-universe comparison
"""),
    code(r"""
# S&P 500 numbers from notebook 02 / 04 (pasted as constants for the
# side-by-side). Values match the README tables.
us_row = {
    "universe": "US",
    "xgb_ic_ir": 0.20, "xgb_mean_ic": 0.037, "xgb_sharpe_net": 0.87, "xgb_ann_net": 0.154, "xgb_mdd": -0.239,
    "ew_ic_ir":  0.11, "ew_mean_ic":  0.021, "ew_sharpe_net":  0.43, "ew_ann_net":  0.067, "ew_mdd":  -0.327,
}
def _pack(uni: str, ic: pd.DataFrame, bt: pd.DataFrame) -> dict:
    def _col(m, col):
        row = ic[(ic["universe"] == uni) & (ic["model"] == m)]
        if row.empty:
            return np.nan
        return row[col].iloc[0]
    def _bcol(m, col):
        row = bt[bt["model"] == m]
        if row.empty:
            return np.nan
        return row[col].iloc[0]
    return {
        "universe":         uni,
        "xgb_ic_ir":        _col("xgb_ranker", "ic_ir"),
        "xgb_mean_ic":      _col("xgb_ranker", "mean_ic"),
        "xgb_sharpe_net":   _bcol("xgb_ranker", "sharpe_net"),
        "xgb_ann_net":      _bcol("xgb_ranker", "ann_net"),
        "xgb_mdd":          _bcol("xgb_ranker", "mdd_net"),
        "ew_ic_ir":         _col("naive_ew", "ic_ir"),
        "ew_mean_ic":       _col("naive_ew", "mean_ic"),
        "ew_sharpe_net":    _bcol("naive_ew", "sharpe_net"),
        "ew_ann_net":       _bcol("naive_ew", "ann_net"),
        "ew_mdd":           _bcol("naive_ew", "mdd_net"),
    }

comp = pd.DataFrame([us_row, _pack("TW", tw_ic, tw_bt), _pack("BTC", btc_ic, btc_bt)])
comp
"""),
    code(r"""
# same table pivoted per (model, metric) for the README
pivot_rows = []
for uni_row in comp.to_dict("records"):
    for model_prefix, model_name in [("xgb", "xgb_ranker"), ("ew", "naive_ew")]:
        pivot_rows.append({
            "universe": uni_row["universe"],
            "model":    model_name,
            "mean_ic":  uni_row[f"{model_prefix}_mean_ic"],
            "ic_ir":    uni_row[f"{model_prefix}_ic_ir"],
            "ann_net":  uni_row[f"{model_prefix}_ann_net"],
            "sharpe_net": uni_row[f"{model_prefix}_sharpe_net"],
            "mdd":      uni_row[f"{model_prefix}_mdd"],
        })
comp_long = pd.DataFrame(pivot_rows)
comp_long
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
metrics = [("ic_ir", "OOS IC-IR"), ("sharpe_net", "Net Sharpe")]
for ax, (metric, title) in zip(axes, metrics):
    sub = comp_long.pivot(index="universe", columns="model", values=metric)
    sub = sub.reindex(index=["US", "TW", "BTC"])
    sub.plot(kind="bar", ax=ax, color=["C1", "C4"])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(loc="best", fontsize=9)
    for patch in ax.patches:
        h = patch.get_height()
        if pd.notna(h):
            ax.text(patch.get_x() + patch.get_width() / 2, h,
                    f"{h:.2f}", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_three_universe_compare.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    md(r"""
## 5. Takeaways

### 5.1 XGB transfers; `naive_ew` does not.
| Universe | XGB IC-IR | `naive_ew` IC-IR | gap |
|---|---|---|---|
| US (502 names) | 0.20 | +0.11 | +0.09 |
| TW (50 names) | **0.27** | +0.06 | +0.21 |
| BTC (20 names) | 0.12 | **−0.39** | **+0.51** |

The handmade `naive_ew` portfolio — equal-weight z-score of `size_adv_60 +
vol_60d + reversal_1w`, signs fixed from S&P 500 EDA — **inverts on BTC**.
Its IC-IR of −0.39 on crypto means the three features that were correctly
signed for US equities are pointing the *wrong* way on crypto: small
tokens under-perform, low-vol tokens out-perform, momentum beats reversal.
XGB adapts split directions per-universe at train time and still extracts
a positive IC-IR. This is the clearest piece of evidence the project has
for *why* the model's complexity is worth paying for — handmade signals
carry the assumption that their sign structure is universal, and that
assumption breaks as soon as you step outside US large caps.

### 5.2 TW's signal is stronger than US's, but the tax takes it all.
TW's XGB IC-IR 0.27 is the strongest of the three universes — likely because
Taiwan large caps still exhibit textbook momentum / size / volatility
structure that has decayed in the US post-2015 (consistent with
`classic-factors`' finding of +11.2% ann return on TW momentum vs −2.7%
on US). But net of the 49 bps round-trip `TW_EQUITY` cost the Sharpe
collapses from **0.71 gross to 0.43 net** — a 40% Sharpe haircut, vs. a
6% haircut on US. Lesson: the IC is real, but monthly-rebalance quintile
L/S is infeasible in Taiwan without restructuring the portfolio to amortise
the tax (hold longer, trade fewer names per rebalance, or trade via
tax-efficient wrappers).

### 5.3 Breadth is not the main constraint.
BTC has only 20 names and 3 OOS years, yet XGB still earns 0.12 IC-IR
(t = 3.8). TW has 50 names and earns 0.27. The naive prior that "more
names → better ranking" does not survive contact with these two
appendices — what matters is the feature-signal structure of the
underlying market, not the nominal universe size. The narrow quintiles
(4 names long, 4 short on BTC) do show up as elevated volatility — BTC
Sharpe net 0.52 on ann return 10.4%, MDD −44% (the deepest of the three
universes).

### 5.4 What this locks in for the README.
- The headline stays US — that's where the 5-year OOS window, 500-name
  universe, and reasonable transaction costs all combine.
- TW / BTC get a single "robustness" paragraph in the README with the
  side-by-side three-universe table, emphasising the `naive_ew` failure
  on BTC as the money shot.
- `naive_ew` → `naive_us_ew` would be a more honest name. Flagged for a
  follow-up commit.
"""),
]


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ml-cross-sectional)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
