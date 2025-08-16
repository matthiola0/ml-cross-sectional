# ml-cross-sectional

Cross-sectional stock ranking on the S&P 500 using LightGBM, with walk-forward
validation and SHAP-based feature attribution.

**Status:** early WIP — scaffolding + data / feature pipeline.

## Research question

Can a gradient-boosted learner systematically rank S&P 500 names by next-month
return better than an equal-weighted classic-factor portfolio, and what is it
actually learning?

## Design at a glance

- **Universe:** current S&P 500 constituents (survivorship bias acknowledged — see
  [Caveats](#caveats)).
- **Target:** forward 1-month return → cross-sectional rank (not absolute).
- **Features (~30):** multi-horizon momentum, realised volatility, RSI, MACD,
  volume z-score, plus classic-factor signals (momentum, reversal, low-vol, ADV
  size). Fundamentals deferred until `qtools` ships a fundamentals loader.
- **Models:** Ridge / Lasso (linear baseline) · LightGBM `LGBMRanker` (primary)
  · XGBoost (robustness check) · equal-weight classic-factor portfolio (naive
  baseline).
- **Validation:** walk-forward, yearly re-train, out-of-sample ≥ 5 years.
- **Robustness:** TW 0050 constituents and BTC top-30 as appendix sections.

Infrastructure (data loaders, backtest engine, factor metrics) comes from
[`qtools`](https://github.com/matthiola0/qtools).

## Repo layout

```
ml-cross-sectional/
├── scripts/
│   ├── download_data.py      # qtools → S&P 500 OHLCV → data/raw/
│   ├── build_features.py     # raw prices → feature matrix → data/processed/
│   └── train.py              # (W6) walk-forward training
├── src/mlcs/
│   ├── features.py           # technical indicators + classic signals
│   ├── model.py              # (W6) model wrappers
│   └── validation.py         # (W6) walk-forward splitter
├── notebooks/
│   ├── 01_feature_eda.ipynb            # (W5)
│   ├── 02_training_walkforward.ipynb   # (W6)
│   ├── 03_shap_analysis.ipynb          # (W7)
│   ├── 04_backtest.ipynb               # (W7)
│   └── 05_robustness_tw_btc.ipynb      # (W7)
└── reports/
```

## Quickstart

```bash
conda create -n ml-cross-sectional python=3.13 -y
conda activate ml-cross-sectional
pip install -e .

python scripts/download_data.py            # S&P 500 OHLCV, 2015-01-01 → 2025-07-31
python scripts/build_features.py           # assemble feature matrix
```

Data is cached under `~/.qtools_cache/` (gitignored); intermediate artifacts
land in `data/` (also gitignored).

## Caveats

- **Survivorship bias.** The universe is *current* S&P 500 membership; names
  that were in the index historically but have since been removed, merged, or
  delisted are invisible. This inflates backtest performance; magnitude is
  discussed in `04_backtest.ipynb` once results land.
- **No fundamentals (yet).** The US feature set is technical-only + classic
  factors until `qtools` ships a fundamentals loader. Tracked in the side-project
  SDD.

## References

- Gu, Kelly, Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
- Lopez de Prado (2018), *Advances in Financial Machine Learning*, chs. 7–8 on
  walk-forward and combinatorial CV.
