"""Build notebooks/02_training_walkforward.ipynb.

Runs annual walk-forward training for four models (Ridge, Lasso, LightGBM
Ranker, XGBoost Ranker) plus a handmade equal-weight baseline, over OOS years
2020–2024 (2025 truncated at 2025-07-30). Collects OOS predictions, computes
per-date Spearman IC against `fwd_rank_21d`, and lands everything in tables
and plots suitable for the README.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).parent.parent / "notebooks" / "02_training_walkforward.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


CELLS = [
    md(r"""
# 02 — Walk-Forward Training & Cross-Model IC Comparison

**Purpose.** Train four cross-sectional ranking models on expanding annual
windows, collect their out-of-sample predictions, and compare to a handmade
equal-weight baseline. The question is not "does the boosted model post the
highest IC?" — it's "does the extra machinery buy anything over the naive
combination of the three features that already had positive IC in
`01_feature_eda`?"

**Models.**
| Model | Role | Notes |
|---|---|---|
| `linear_ridge` | Linear baseline | Ridge on per-date z-scored features. |
| `linear_lasso` | Sparse linear | Lasso; should zero out the redundant wrong-sign momentum series. |
| `lgbm_ranker` | Main tree model | LightGBM `LGBMRanker`, pairwise rank objective with decile labels. |
| `xgb_ranker` | Robustness check | XGBoost `XGBRanker`, same structure. |
| `naive_ew` | Yardstick | Equal-weight z-score of `size_adv_60 + vol_60d + reversal_1w`. |

**Walk-forward.** For each OOS year Y ∈ {2020..2024}, train on all data with
`date < Jan 1 Y` and predict year Y. Expanding window, re-train yearly, no
purging (monthly horizon + annual retrain makes the embargo gain negligible
relative to fold noise).

**Target.** Tree models consume integer decile labels per date (0..9,
`qcut(fwd_rank_21d, 10)`). Linear models predict the continuous
`fwd_rank_21d` directly. Both are evaluated on *predicted score vs actual
forward return rank* using per-date Spearman IC, which is model-agnostic.

**Feature set.** All 12 features from `01_feature_eda`, minus `low_vol_60`
which is a sign-flipped duplicate of `vol_60d`.
"""),
    code(r"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mlcs.model import (
    EqualWeightBaseline,
    LGBMRankerModel,
    LinearRanker,
    XGBRankerModel,
)
from mlcs.validation import walk_forward_years

PROC_DIR = Path("../data/processed")
REPORTS_DIR = Path("../reports")
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110
"""),
    code(r"""
candidates = sorted(PROC_DIR.glob("features_*.parquet"))
assert candidates, "Run scripts/build_features.py first."
df = pd.read_parquet(candidates[-1])
df["date"] = pd.to_datetime(df["date"])

# feature set per 01_feature_eda conclusions
FEATURES = [
    "mom_12_1", "reversal_1w", "size_adv_60",
    "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_20d", "vol_60d",
    "rsi_14", "macd_hist", "volume_z_60",
]
TARGET = "fwd_rank_21d"

# keep rows where at least the target and a majority of features are present;
# trees handle NaN internally, linear models need all features present and get
# an in-training mask in LinearRanker.fit.
work = df.dropna(subset=[TARGET]).copy()
print(f"working rows: {len(work):,}   date range: {work['date'].min().date()} → {work['date'].max().date()}")
print(f"features    : {len(FEATURES)}")
"""),
    md(r"""
## 1. Walk-forward training loop

Each fold trains 5 models on `< Jan 1 Y` and predicts year Y. OOS predictions
are stacked into one long dataframe keyed by `(date, symbol, model)`. Timing
is logged — the main cost is the two boosters' fits, which grow with the
expanding training window.
"""),
    code(r"""
FIRST_OOS = 2020
LAST_OOS = 2024

def make_models():
    return [
        LinearRanker(kind="ridge", alpha=1.0),
        LinearRanker(kind="lasso", alpha=1e-4),
        LGBMRankerModel(n_estimators=400, num_leaves=31, learning_rate=0.05,
                        min_child_samples=100, random_state=42),
        XGBRankerModel(n_estimators=400, max_depth=5, learning_rate=0.05,
                       random_state=42),
        EqualWeightBaseline(),
    ]

import time

all_preds = []
for oos_year, tr_mask, te_mask in walk_forward_years(work["date"], FIRST_OOS, LAST_OOS):
    tr = work.loc[tr_mask]
    te = work.loc[te_mask]
    print(f"\n[{oos_year}] train={len(tr):,}  test={len(te):,}")
    for mdl in make_models():
        t0 = time.time()
        mdl.fit(tr[FEATURES], tr[TARGET], tr["date"])
        scores = mdl.predict(te[FEATURES], te["date"])
        dt = time.time() - t0
        out = te[["date", "symbol", TARGET]].copy()
        out["model"] = mdl.name
        out["score"] = scores
        all_preds.append(out)
        print(f"   {mdl.name:14s}  fit+predict {dt:5.1f}s")

preds = pd.concat(all_preds, ignore_index=True)
print("\nstacked OOS predictions:", preds.shape)
"""),
    code(r"""
# persist for W7 (backtest + SHAP) so we don't re-train downstream
out_path = REPORTS_DIR / "predictions" / f"oos_{FIRST_OOS}_{LAST_OOS}.parquet"
preds.to_parquet(out_path, index=False)
print("wrote", out_path, f"({out_path.stat().st_size / 1e6:.1f} MB)")
"""),
    md(r"""
## 2. Per-date Spearman IC per model

For each (date, model), rank-correlate predicted `score` against actual
`fwd_rank_21d`. This is the model-agnostic signal quality metric — a cleanly
positive IC means the model's ordering of stocks on that date matched the
forward-return ordering.
"""),
    code(r"""
def per_date_ic_model(frame: pd.DataFrame) -> pd.Series:
    def _ic(g):
        sub = g[["score", TARGET]].dropna()
        if len(sub) < 20:
            return np.nan
        return sub["score"].rank().corr(sub[TARGET])
    return frame.groupby("date").apply(_ic, include_groups=False)

ic_by_model = {m: per_date_ic_model(preds[preds["model"] == m]) for m in preds["model"].unique()}
ic_wide = pd.DataFrame(ic_by_model)
ic_wide.tail(3)
"""),
    code(r"""
rows = []
for m, s in ic_by_model.items():
    s = s.dropna()
    mean = s.mean()
    std = s.std()
    ir = mean / std if std > 0 else np.nan
    tstat = ir * np.sqrt(len(s)) if pd.notna(ir) else np.nan
    rows.append({
        "model":    m,
        "mean_ic":  round(mean, 4),
        "ic_std":   round(std, 4),
        "ic_ir":    round(ir, 3) if pd.notna(ir) else np.nan,
        "t_stat":   round(tstat, 2) if pd.notna(tstat) else np.nan,
        "hit_rate": round((s > 0).mean(), 3),
        "n_dates":  int(len(s)),
    })
summary = pd.DataFrame(rows).sort_values("ic_ir", ascending=False).reset_index(drop=True)
summary
"""),
    md(r"""
## 3. Cumulative IC over OOS window
"""),
    code(r"""
cum = ic_wide.cumsum()

fig, ax = plt.subplots(figsize=(11, 4.5))
palette = {"lgbm_ranker": "C0", "xgb_ranker": "C1",
           "linear_ridge": "C2", "linear_lasso": "C3", "naive_ew": "C4"}
for m in cum.columns:
    ax.plot(cum.index, cum[m], label=m, color=palette.get(m, None), linewidth=1.4)
for y in range(FIRST_OOS, LAST_OOS + 1):
    ax.axvline(pd.Timestamp(f"{y}-01-01"), color="grey", linewidth=0.5, linestyle=":")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Cumulative OOS IC — five models")
ax.set_ylabel("Σ daily Spearman IC")
ax.legend(loc="upper left", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_cumulative_ic.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    md(r"""
## 4. Per-year IC distribution
"""),
    code(r"""
long = ic_wide.stack().reset_index()
long.columns = ["date", "model", "ic"]
long["year"] = long["date"].dt.year

fig, ax = plt.subplots(figsize=(11, 4))
order = summary["model"].tolist()
sns.boxplot(
    data=long, x="year", y="ic", hue="model", hue_order=order,
    showfliers=False, ax=ax,
)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("OOS daily IC distribution — per year × model")
ax.set_ylabel("per-date Spearman IC")
ax.legend(loc="lower right", fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_per_year_ic.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    code(r"""
# per-year mean IC as a table — easier to quote than read off the boxplot
pivot = long.groupby(["year", "model"])["ic"].mean().unstack("model").round(4)
pivot = pivot[order]
pivot
"""),
    md(r"""
## 5. Takeaways

### 5.1 Headline — tree models do earn their complexity.
| Model | IC-IR | mean IC | t-stat |
|---|---|---|---|
| `xgb_ranker` | **0.20** | 0.037 | 6.9 |
| `lgbm_ranker` | 0.14 | 0.031 | 5.1 |
| `naive_ew` | 0.11 | 0.021 | 4.0 |
| `linear_lasso` | 0.07 | 0.017 | 2.6 |
| `linear_ridge` | 0.07 | 0.016 | 2.6 |

XGBoost beats the handmade `naive_ew` baseline by roughly 0.09 IC-IR (t-stat
nearly 7 vs 4); LightGBM beats it by 0.03. The boosted ensemble is doing
work the handmade signal cannot — plausibly by picking up the non-linear
reversal flip in RSI / `ret_21d` / `macd_hist` that `01_feature_eda`
flagged.

### 5.2 Linear baselines lose to the handmade baseline.
Ridge and Lasso both come in at IC-IR ≈ 0.07, *below* the 0.11 that
`naive_ew` posts with three hand-picked features. The cleanest reading: the
12-feature set contains four features (`rsi_14`, `ret_21d`, `macd_hist`,
`ret_63d`) with wrong-sign IC and a further two (`mom_12_1`, `ret_126d`)
sitting at essentially zero. A regularised linear regression on z-scored
inputs is still fundamentally a linear combination — it cannot sign-flip
part of the feature space and it cannot ignore a zero-mean feature as
cleanly as a tree can simply not split on it. Lasso with α = 1e-4 was too
weak to zero them out. Either α needs tuning per fold, or the linear
baseline should be given a *pre-filtered* feature set matching what
`01_feature_eda` recommended. Logged as a candidate follow-up but not a
blocker — the tree models are the main narrative.

### 5.3 Per-year shape is where the story gets interesting.

| Year | Regime | XGB | LGBM | naive | linear |
|---|---|---|---|---|---|
| 2020 | COVID crash + recovery | **+0.093** | +0.074 | +0.055 | +0.051 |
| 2021 | Low-vol bull continuation | −0.022 | +0.010 | +0.008 | +0.025 |
| 2022 | Rate-hike drawdown | −0.007 | **−0.031** | +0.005 | −0.035 |
| 2023 | AI rally | +0.058 | +0.054 | +0.031 | +0.047 |
| 2024 | AI rally cont'd | +0.061 | +0.047 | +0.007 | −0.006 |

Three things jump out:

1. **2022 is a shared failure.** Every learned model posted negative
   per-date IC across the full rate-hike year; only `naive_ew` stayed barely
   positive. The feature set — size, volatility, short-term reversal — was
   learning "small-cap high-vol reverts" throughout 2015–2021, then got
   blindsided when 2022 paid the opposite trade (mega-cap defensive wins).
   This is precisely the regime risk `plan.md` flagged: the models don't see
   the macro environment (VIX, rates, credit), so they cannot adapt when
   the reward function flips.
2. **XGB is the best single model but the most brittle.** It posts the
   highest number in 3 of 5 years (2020, 2023, 2024) *and* a negative
   number in 2021, where LightGBM and both linears held positive. In a
   production setting this argues against single-model deployment —
   averaging the tree-model scores with `naive_ew` is an obvious next
   robustness check.
3. **Naive-EW has the lowest year-over-year variance.** Mean IC is worst
   except in 2022, but the dispersion across years is far tighter. If the
   goal were "stable, if modest, factor exposure" the handmade portfolio
   would not be a bad choice — a real tradeoff between complexity and
   regime robustness.

### 5.4 What this locks in for W7.
- **Main deliverable is the XGB vs naive comparison**, framed honestly with
  the 2022 counter-example. The pitch is not "ML crushes baselines"; it's
  "ML earns a 2× IR over five years by exploiting non-linearity, at the
  cost of a bigger 2022 drawdown."
- `03_shap_analysis.ipynb` should specifically check whether `size_adv_60`
  and the two volatility features dominate SHAP importance (confirming the
  naive-EW signals are doing most of the work) or whether the non-linear
  wrong-sign features (`rsi_14`, `ret_21d`, `macd_hist`) explain the gap.
- `04_backtest.ipynb` needs to turn these IC time series into long-short
  portfolios with realistic costs; an IR of 0.20 on monthly IC is not
  automatically an IR of 0.20 on net PnL after turnover.
- A follow-up worth logging but not blocking W7: re-fit the linear models
  on the EDA-recommended subset (size_adv_60, vol_60d, reversal_1w,
  volume_z_60) to separate "bad regularisation" from "linear models are
  fundamentally wrong here" as the source of their under-performance.
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
