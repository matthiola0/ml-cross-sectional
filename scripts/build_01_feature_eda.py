"""Build notebooks/01_feature_eda.ipynb from source cells defined here.

Keeps notebook content as Python source so diffs are readable. Re-run whenever
the analysis text/code changes.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).parent.parent / "notebooks" / "01_feature_eda.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


CELLS = [
    md(r"""
# 01 — Feature EDA

**Purpose.** Before touching any model, sanity-check the feature panel that
`scripts/build_features.py` produced: coverage, distributional shape, how much
information overlap there is between features, and how each one scores on a
single-feature IC baseline against the 21-day forward cross-sectional rank.

**Sample.** 502 current S&P 500 constituents, 2015-01-02 → 2025-07-30.
Survivorship bias acknowledged — see repo README.

**What a "useful" feature looks like here.**
- Reasonable coverage (not 80% NaN outside an initial warm-up period).
- Spearman rank IC against `fwd_rank_21d` with mean > 0 and IC-IR > ~0.1.
- Not a near-duplicate of another feature already in the set (cross-sectional
  corr below ~0.9 on average).

IC values will be modest — a single monthly-horizon feature on liquid US
large-caps rarely clears IC = 0.05. The model's job is to *combine* them.
"""),
    code(r"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROC_DIR = Path("../data/processed")
FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 110
"""),
    code(r"""
# pick the newest feature parquet
candidates = sorted(PROC_DIR.glob("features_*.parquet"))
assert candidates, "Run scripts/build_features.py first."
path = candidates[-1]
print("loading", path.name)

df = pd.read_parquet(path)
df["date"] = pd.to_datetime(df["date"])
FEATURE_COLS = [c for c in df.columns if c not in (
    "date", "symbol", "fwd_ret_21d", "fwd_rank_21d",
)]
print(f"rows: {len(df):,}   features: {len(FEATURE_COLS)}")
df.head(3)
"""),
    md(r"""
## 1. Coverage

How many symbols do we have at each date, and how much of each feature survives
the warm-up period?
"""),
    code(r"""
per_date = df.groupby("date")["symbol"].nunique()
print(f"date range: {per_date.index.min().date()} → {per_date.index.max().date()}")
print(f"symbols/date — median: {per_date.median():.0f}   min: {per_date.min()}   max: {per_date.max()}")

fig, ax = plt.subplots(figsize=(10, 2.8))
per_date.plot(ax=ax, color="steelblue")
ax.set_title("Symbols available per date")
ax.set_ylabel("count")
ax.set_xlabel("")
plt.tight_layout()
plt.show()
"""),
    code(r"""
# feature non-NaN coverage as a % of total rows, and as a % of rows after an
# arbitrary 300-day warmup (ret_252d's warmup dominates otherwise)
warmup_cutoff = df["date"].unique()
warmup_cutoff = np.sort(warmup_cutoff)[300]
post = df[df["date"] >= warmup_cutoff]

coverage = pd.DataFrame({
    "all":          df[FEATURE_COLS].notna().mean(),
    "post_warmup":  post[FEATURE_COLS].notna().mean(),
}).round(3)
coverage.sort_values("post_warmup")
"""),
    md(r"""
## 2. Distributions

Each feature over the full panel. Features with heavy tails (returns,
`size_adv_60`) are plotted on a signed-log axis so the bulk of the mass is
visible.
"""),
    code(r"""
def _sym_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

n = len(FEATURE_COLS)
ncols = 3
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.3 * nrows))
axes = axes.flatten()

SIGNED_LOG = {"size_adv_60"}

for ax, col in zip(axes, FEATURE_COLS):
    vals = df[col].dropna()
    if col in SIGNED_LOG:
        ax.hist(_sym_log(vals), bins=80, color="steelblue")
        ax.set_title(f"{col}  (sign·log1p)")
    else:
        # clip to 0.5–99.5 percentile so a few outliers don't flatten the shape
        lo, hi = vals.quantile([0.005, 0.995])
        ax.hist(vals.clip(lo, hi), bins=80, color="steelblue")
        ax.set_title(col)
    ax.set_yticks([])

for ax in axes[n:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_feature_distributions.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    md(r"""
## 3. Feature overlap

How correlated are the features, *cross-sectionally*? A feature set full of
near-duplicates gives a linear model collinearity headaches and gives a tree
model redundant splits.

Method: for each date, compute the cross-sectional Spearman correlation
between every feature pair, then average across dates. This matches how the
downstream ranker will "see" the features — relative ordering within a
single day, not time-series levels.
"""),
    code(r"""
def mean_crosssectional_corr(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    acc = np.zeros((len(cols), len(cols)))
    n_dates = 0
    for _, g in data.groupby("date"):
        sub = g[cols].dropna(how="any")
        if len(sub) < 20:
            continue
        acc += sub.rank().corr().values
        n_dates += 1
    return pd.DataFrame(acc / n_dates, index=cols, columns=cols)

corr = mean_crosssectional_corr(df, FEATURE_COLS)

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "mean CS Spearman"})
ax.set_title("Mean cross-sectional Spearman correlation between features")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_feature_corr.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    code(r"""
# flag near-duplicate pairs (|ρ| > 0.9) for the notebook write-up
pairs = []
for i, a in enumerate(FEATURE_COLS):
    for b in FEATURE_COLS[i + 1:]:
        r = corr.loc[a, b]
        if abs(r) > 0.9:
            pairs.append((a, b, round(float(r), 3)))
pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"])
"""),
    md(r"""
## 4. Single-feature IC

For each feature, per-date cross-sectional Spearman rank correlation with
`fwd_rank_21d` → gives an IC time series. Summary stats:

- **mean IC** — average directional edge per date (positive = useful).
- **IC-IR** — mean / std (monthly-ish risk-adjusted edge).
- **t-stat** — IC-IR × √N (informal significance; IC is serially correlated so
  treat as heuristic, not a p-value).
- **hit rate** — fraction of dates with IC > 0.
"""),
    code(r"""
def per_date_ic(data: pd.DataFrame, feature: str, target: str = "fwd_rank_21d") -> pd.Series:
    def _corr(g):
        sub = g[[feature, target]].dropna()
        if len(sub) < 20:
            return np.nan
        return sub[feature].rank().corr(sub[target])
    return data.groupby("date").apply(_corr, include_groups=False)

ic_frames = {}
for col in FEATURE_COLS:
    ic_frames[col] = per_date_ic(df, col)
ic_df = pd.DataFrame(ic_frames)
ic_df.tail(3)
"""),
    code(r"""
rows = []
for col in FEATURE_COLS:
    s = ic_df[col].dropna()
    if len(s) == 0:
        continue
    mean = s.mean()
    std = s.std()
    ir = mean / std if std > 0 else np.nan
    tstat = ir * np.sqrt(len(s)) if pd.notna(ir) else np.nan
    rows.append({
        "feature":    col,
        "mean_ic":    round(mean, 4),
        "ic_std":     round(std, 4),
        "ic_ir":      round(ir, 3) if pd.notna(ir) else np.nan,
        "t_stat":     round(tstat, 2) if pd.notna(tstat) else np.nan,
        "hit_rate":   round((s > 0).mean(), 3),
        "n_dates":    int(len(s)),
    })
ic_summary = pd.DataFrame(rows).sort_values("ic_ir", ascending=False).reset_index(drop=True)
ic_summary
"""),
    code(r"""
# cumulative IC — flat-trending features are signals that persist; jagged
# cumulative paths mean the factor only works in specific regimes.
top = ic_summary.head(8)["feature"].tolist()
cum = ic_df[top].cumsum()

fig, ax = plt.subplots(figsize=(11, 4.5))
for col in top:
    ax.plot(cum.index, cum[col], label=col, linewidth=1.3)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Cumulative IC — top-8 features by IC-IR")
ax.set_ylabel("Σ daily IC")
ax.legend(loc="best", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_feature_cumulative_ic.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    md(r"""
## 5. Takeaways

### 5.1 Coverage is not a constraint.
Every feature is ≥ 99% populated after the 300-day warm-up window. The only
pre-warm-up gaps trace to `ret_252d` and `mom_12_1`, which mechanically need
252 + 21 days of history. Nothing to drop on coverage grounds.

### 5.2 `low_vol_60` and `vol_60d` are the same feature.
Their mean cross-sectional Spearman is **−0.999**. That is by construction —
`low_vol_60` is the pct-return rolling std sign-flipped, `vol_60d` is the
log-return rolling std raw, and at daily frequency log-return ≈ pct-return to
three decimals. **Action: drop `low_vol_60` from the model feature set**
before W6. Tree splits would be unaffected but (a) SHAP attributions would
split the signal's importance arbitrarily across the pair, and (b) the linear
baselines (Ridge / Lasso) get a nasty collinearity hit. Keep `vol_60d` —
raw-level volatility is the more natural axis to let the model sign itself.

### 5.3 The single-feature edge is where it should be, and where it should *not*.

**Signals that earn their place** (IC-IR > 0.1, t-stat > 5):
- `size_adv_60` (IC-IR **+0.16**): the strongest single factor. Small-ADV
  names out-rank large-ADV names — classic liquidity / size premium, still
  alive in S&P 500 after 2015. This is the feature the model is most likely
  to lean on.
- `vol_60d` and `vol_20d` (IC-IR ≈ +0.12): **high-vol names outperform**. The
  textbook "low-vol anomaly" is *reversed* in this sample. Plausibly an
  artefact of (a) post-2015 growth / high-beta leadership or (b) survivorship
  bias — failed high-vol names are missing from today's constituent list.
  Worth re-testing in W6 on a sector-neutral basis.
- `reversal_1w` (IC-IR +0.08): short-horizon mean reversion survives.

**Signals that are dead** (|IC-IR| < 0.05):
- `mom_12_1`: mean IC effectively zero (−0.002, t = −0.5). The textbook 12-1
  momentum factor has *decayed to noise* on S&P 500 post-2015. This matches
  the Repo 1 `classic-factors` narrative exactly — useful corroboration.
- `ret_126d`, `volume_z_60`: weak / indeterminate.

**Signals with the "wrong" sign vs. textbook** (significantly negative IC):
- `rsi_14` (IC-IR −0.10), `ret_21d` (−0.09), `macd_hist` (−0.06), `ret_63d`
  (−0.08): all conventional *momentum-direction* features come in negative at
  the one-month forward horizon. The honest reading is that on modern
  large-cap US, these are reading as *reversal* signals, not momentum. A tree
  model will pick the right split direction automatically; the linear
  baselines will if we include the sign-swapped versions. What *not* to do:
  silently flip the sign just because it looks better on IC — keep them raw
  and let regularised weights speak.

### 5.4 Regime shape matters more than headline IC.
The cumulative-IC panel is the chart to squint at before trusting any
headline number. Relevant questions:
- Does `size_adv_60`'s +0.16 IC come from one structural drift (steady
  upslope) or a few regime windows (step-up around COVID, flat elsewhere)?
- Does `vol_60d`'s positive IC change sign around the 2022 rate-hike
  drawdown? If yes, that's a regime-dependent signal and the model will need
  regime features or rolling retraining to use it safely.

W6 walk-forward should partition the OOS window the same way Repo 1 did
(2020 COVID, 2021–22 rate hike, 2023–25 AI rally) so we can quote per-regime
model performance, not just a pooled number.

### 5.5 Open questions going into modelling.
- **Size × vol interaction.** The two strongest signals are small-ADV and
  high-vol. Are they orthogonal, or is the model just re-learning "small,
  volatile" as one axis? A 2×2 quintile table (size × vol, mean forward
  return) before training will clarify.
- **Momentum residualisation.** Given `mom_12_1` ≈ 0 IC, it is probably
  *not* worth residualising other features against it. The classic Grinold
  argument for doing so assumes a dominant momentum factor — here there
  isn't one. Skip residualisation; let the ranker handle interactions.
- **Feature set for the linear baseline.** Likely keep `size_adv_60`,
  `vol_60d`, `reversal_1w`, `volume_z_60`, and drop the short-horizon
  returns + RSI (they are informative only after a sign flip and collide
  with `reversal_1w`). Tree model keeps the full set minus `low_vol_60`.
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
