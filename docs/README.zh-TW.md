# ml-cross-sectional

> **Languages**: [English](../README.md) · **繁體中文**

S&P 500 上的 LightGBM / XGBoost 橫斷面選股模型，含 walk-forward 驗證、SHAP 歸因、扣成本 long-short 回測。

研究框架：[`qtools`](../../qtools)。對照研究：[`classic-factors`](../../classic-factors)。

![net equity curve](../reports/figures/04_net_equity.png)

## TL;DR

`XGBRanker` 模型 + 12 個價量特徵，2020–2024 OOS 在 S&P 500 quintile L/S 上達 **15.4% 年化、Net Sharpe 0.87、MDD −24%** —— 已扣 5 bps 單邊成本。約是「EDA 階段挑出 IC-IR 為正的 3 個特徵」等權 baseline 的 2× Sharpe / 2× 年化報酬。誠實 caveat：股票池有存活者偏差（當前 S&P 500）、未做 sector / beta 中性化、2022 升息 regime 把所有訊號（含 baseline）都打殘。

## 研究問題

gradient-boosted 橫斷面 ranker 能否 *系統性* 擊敗一個僅用「`01_feature_eda` 找出單因子 IC 為正的特徵子集」做的等權組合？扣完交易成本、跨 regime 都成立嗎？SHAP 能否解釋這個差距，讓故事不是「ML 是魔術」而是「ML 抓到線性模型本質上抓不到的條件效果」？

## 範圍

| | |
|---|---|
| 股票池 | 502 檔當前 S&P 500 成分股 |
| 期間 | 2015-01-02 → 2025-07-30（train + OOS）|
| OOS 視窗 | 2020 → 2024（5 年、1,258 個交易日）|
| 預測 target | Forward 21 日報酬的橫斷面 rank |
| 特徵 | 12 個價量特徵 — 3 個經典因子訊號 + 4 個多視窗報酬 + 2 個 realised vol + RSI + MACD + 成交量 z-score |
| 模型 | Ridge · Lasso · LightGBM `LGBMRanker` · XGBoost `XGBRanker` · 手刻等權 baseline |
| 驗證 | 年度 expanding-window walk-forward，無 purge |
| 成本模型 | `qtools.backtest.costs.US_EQUITY`（1 bp 手續費 + 4 bps 滑點 = 單邊 5 bps）|

`plan.md` 列出的基本面特徵（P/B、P/E、ROE）刻意 out-of-scope：`qtools` 還沒 fundamentals loader，本研究限於 OHLCV 可推導的訊號。

## 結果

### 特徵 EDA — [notebook 01](../notebooks/01_feature_eda.ipynb)

單因子 Spearman IC vs 21 日 forward rank（2015 → 2025）。前段與後段示範如下，全表見 notebook：

| 特徵 | mean IC | IC-IR | t-stat | 解讀 |
|---|---|---|---|---|
| `size_adv_60` | +0.016 | **+0.16** | 8.1 | 小 ADV 溢酬在 S&P 500 仍存在 |
| `vol_60d` | +0.029 | +0.12 | 6.2 | **低波異象反轉** — 高波勝出 |
| `vol_20d` | +0.025 | +0.12 | 5.9 | 同上 |
| `reversal_1w` | +0.013 | +0.08 | 3.9 | 短期反轉仍有效 |
| `mom_12_1` | −0.002 | −0.01 | −0.5 | **12-1 動量 post-2015 已衰減成雜訊** |
| `rsi_14` | −0.016 | −0.10 | −5.0 | 表現如 *反轉* 而非動量 |
| `low_vol_60` | −0.029 | −0.12 | −6.1 | 與 `vol_60d` ρ = −0.999（反向 duplicate，已剔除） |

12-1 動量這個發現獨立重現 `classic-factors` Repo 1 的結論。

### Walk-forward 訓練 — [notebook 02](../notebooks/02_training_walkforward.ipynb)

逐日 OOS Spearman IC vs fwd 21 日 rank，pooled 2020–2024：

| 模型 | mean IC | IC-IR | t-stat | hit rate |
|---|---|---|---|---|
| `xgb_ranker` | 0.037 | **0.20** | 6.9 | 56.7% |
| `lgbm_ranker` | 0.031 | 0.14 | 5.1 | 54.5% |
| `naive_ew` | 0.021 | 0.11 | 4.0 | 52.6% |
| `linear_lasso` | 0.017 | 0.07 | 2.6 | 49.0% |
| `linear_ridge` | 0.016 | 0.07 | 2.6 | 49.6% |

逐年 IC 暴露 regime 故事 — 2022 整年只有 `naive_ew` 撐住：

| 年份 | XGB | LGBM | naive | lasso | ridge |
|---|---|---|---|---|---|
| 2020（COVID）| **+0.093** | +0.074 | +0.055 | +0.052 | +0.051 |
| 2021（low-vol bull）| −0.022 | +0.010 | +0.008 | +0.025 | +0.025 |
| 2022（升息）| −0.007 | −0.031 | +0.005 | −0.035 | −0.035 |
| 2023（AI rally）| +0.058 | +0.054 | +0.031 | +0.047 | +0.045 |
| 2024（AI rally）| +0.061 | +0.047 | +0.007 | −0.006 | −0.006 |

### SHAP 歸因 — [notebook 03](../notebooks/03_shap_analysis.ipynb)

TreeExplainer 跑 2024 OOS 抽樣 10,080 row。Mean |SHAP| 排名前段：

| 排名 | 特徵 | mean \|SHAP\| | 在 `naive_ew`？ |
|---|---|---|---|
| 1 | `size_adv_60` | 0.149 | ✓ |
| 2 | `vol_60d` | 0.087 | ✓ |
| 3 | `ret_126d` | **0.065** | ✗ |
| 4 | `ret_252d` | 0.046 | ✗ |
| 5 | `mom_12_1` | 0.029 | ✗ |
| 11 | `reversal_1w` | 0.005 | ✓ |
| 12 | `volume_z_60` | 0.001 | ✗ |

約 55% 全局 |SHAP| 落在 `naive_ew` 的 3 個特徵；其餘 9 個分到 45%。`ret_126d` SHAP 排第 3，但單因子 IC-IR −0.03 —— 它本身沒有預測力，但 tree 在 size 與 vol 桶下條件性使用它。這就是 tree model 嚴格優於 linear model 的教科書場景。

### 扣成本回測 — [notebook 04](../notebooks/04_backtest.ipynb)

Quintile L/S、dollar-neutral、月頻、`US_EQUITY` 成本（單邊 5 bps）。OOS 2020–2024：

| 模型 | 年化淨報酬 | Net Sharpe | MDD | 平均 turnover | 年成本拖累 |
|---|---|---|---|---|---|
| `xgb_ranker` | **15.4%** | **0.87** | **−23.9%** | 159% | 94 bps |
| `lgbm_ranker` | 13.1% | 0.68 | −28.0% | 159% | 94 bps |
| `naive_ew` | 6.7% | 0.43 | −32.7% | 231% | 136 bps |
| `linear_lasso` | 5.3% | 0.34 | −33.1% | 218% | 129 bps |
| `linear_ridge` | 5.2% | 0.34 | −33.2% | 221% | 130 bps |

兩個值得注意的反直覺發現（vs「ML 交易成本貴」這個 prior）：

- Tree 模型 **turnover 反而較低**（159%）vs 手刻 baseline（231%）。Threshold split 定義的 decile 邊界，比每月重新 z-score 三個特徵和的成員洗牌少。
- Tree 模型 **MDD 也較小**（XGB −24%）vs baseline（−33%）。ML quintile 在多空兩腿都更分散，2022 集中爆雷被緩衝。

逐年 Net Sharpe：

| 年份 | XGB | LGBM | naive | lasso | ridge |
|---|---|---|---|---|---|
| 2020 | **1.10** | 1.01 | 0.42 | 0.46 | 0.44 |
| 2021 | 0.38 | 0.64 | 0.82 | **0.90** | 0.93 |
| 2022 | −0.39 | −0.51 | **−0.41** | −0.73 | −0.70 |
| 2023 | **2.99** | 2.36 | 1.99 | 2.12 | 2.08 |
| 2024 | **1.34** | 0.77 | 0.23 | −0.21 | −0.19 |

### 跨股票池 robustness — [notebook 05](../notebooks/05_robustness_tw_btc.ipynb)

相同 12 個特徵、相同 XGB ranker，重訓於台灣 0050（50 檔）與 hard-coded 20 檔 USDT pairs（Binance，2018–2025、OOS 2022–2024）。

| 股票池 | XGB IC-IR | XGB Net Sharpe | `naive_ew` IC-IR | `naive_ew` Net Sharpe |
|---|---|---|---|---|
| 美股（502 檔，2020–2024）| +0.20 | +0.87 | +0.11 | +0.43 |
| 台股 0050（50 檔，2020–2024）| **+0.27** | +0.43 | +0.06 | +0.18 |
| BTC-uni（20 檔，2022–2024）| +0.12 | +0.52 | **−0.39** | **−0.25** |

關鍵觀察是 BTC 上的 `naive_ew` 失敗：3 個手刻特徵（小 ADV long、高波 long、短期反轉 long）在加密的符號 *是反的* —— 小幣 underperform、低波勝出、動量勝過反轉。XGB ranker 訓練時自己學到符號翻轉，仍交出正 IC-IR。**股票池大小（20 vs 500）不重要，重要的是底層市場的符號結構**。

次要發現：台股 IC-IR（+0.27）反而是 3 個股票池 **最強**，但 49 bps 來回 `TW_EQUITY` 成本（vs 美股單邊 5 bps）把 Sharpe 從 gross 0.71 砍到 net 0.43 —— 美股只砍 6%，台股砍 40%。訊號是真的，但台股成本結構讓月頻 quintile L/S 不可行。

## 失效模式

1. **2022 是所有模型共同的負 Sharpe regime**。升息 drawdown 反向打小型股 / 高波 exposure（5 個訊號共同 load 的主軸），特徵集裡沒有任何宏觀環境感知。對照 Repo 5（`ml-return-forecast`，未建）正是針對此設計，加 macro 特徵（VIX、殖利率、信用利差）。

2. **XGB 在 2021 輸給線性 baseline**：Net Sharpe 0.38 vs 0.90。在平滑延續 regime，COVID 期間訓練的 tree splits 顯然對 2020 反轉 over-fit、漏掉 2021 延續。一個 fold-weighted XGB + Lasso ensemble 應該比單獨 XGB robust。

3. **線性 baseline 輸給 `naive_ew`**：IC-IR 0.07 vs 0.11。12 個特徵裡 4 個 IC 符號是反的（`rsi_14`、`ret_21d`、`macd_hist`、`ret_63d`），tree 可以乾脆不分裂，但 regularised linear regression 必須給係數。Lasso α = 1 × 10⁻⁴ 太弱沒歸零。這是可修的失敗 —— 線性 baseline 應預先用 IC-IR > 0 的子集，已記在 SDD 跟進。

## 限制

- **存活者偏差**：股票池為當前 S&P 500，2020–2024 期間被剔除的不可見。Net Sharpe 估計被高估 10–20%（量級而非實測）。
- **無 sector / beta 中性化**：Quintile L/S 在 2023 應該重壓 high-beta tech 然後賺到。Sector-neutral 版本應 2023 Sharpe 較低、2022 較不負。
- **無 short borrow cost**：典型 GC borrow 50–100 bps/年，會直接從淨報酬扣。
- **未模擬 point-in-time 成分股**：2022 才加入指數的名字 2015 也有權重 —— 第二種 look-ahead，疊在存活者偏差之上。

## 結構

```
ml-cross-sectional/
├── src/mlcs/
│   ├── features.py             # 技術 + 經典因子訊號
│   ├── model.py                # 5 個 ranker 共用 fit/predict 介面
│   └── validation.py           # 年度 walk-forward splitter
├── scripts/
│   ├── download_data.py                 # qtools → data/raw/
│   ├── build_features.py                # → data/processed/features_*.parquet
│   ├── build_01_feature_eda.py          # notebook source（重執行可重生 ipynb）
│   ├── build_02_training_walkforward.py
│   ├── build_03_shap_analysis.py
│   ├── build_04_backtest.py
│   └── build_05_robustness_tw_btc.py
├── notebooks/
│   ├── 01_feature_eda.ipynb
│   ├── 02_training_walkforward.ipynb
│   ├── 03_shap_analysis.ipynb
│   ├── 04_backtest.ipynb
│   └── 05_robustness_tw_btc.ipynb
└── reports/
    ├── figures/                # 入 git；README + notebook 引用
    └── predictions/            # OOS score parquet 給 03 / 04 用
```

## 重現

```bash
# clone + install（qtools 透過 pyproject.toml 自動拉）
git clone https://github.com/matthiola0/ml-cross-sectional
cd ml-cross-sectional
conda create -n ml-cross-sectional python=3.13 -y
conda activate ml-cross-sectional
pip install -e .

# 1. 填價格快取（與 classic-factors 共用；若該 repo 已跑過則 cache hit）
python scripts/download_data.py                # 首次 ~5 分鐘，cached 即時

# 2. 建特徵矩陣
python scripts/build_features.py               # → data/processed/features_*.parquet

# 3. 重執行 notebooks（walk-forward 訓練 ~4 分鐘；其餘 < 1 分鐘）
python -m ipykernel install --user --name ml-cross-sectional
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=ml-cross-sectional \
    --inplace notebooks/*.ipynb
```

Notebook source 可由 `scripts/build_*.py` 重生 —— 編輯後重跑、再執行 ipynb。

**本地開發**：若有 [`qtools`](https://github.com/matthiola0/qtools) 本地 clone 想邊改邊用而不 push，安裝後執行 `pip install -e ../qtools` 覆寫 git-installed 版本為 editable 本地版本。

## 參考文獻

**橫斷面 ML in 資產定價**
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *Review of Financial Studies*, 33(5), 2223–2273.
  [doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009) — 線性 / tree / 神經網路在月頻美股報酬預測的 benchmark 比較。本 repo **不是** 直接複製 —— 我們 (i) 用 `LGBMRanker` / `XGBRanker` 預測橫斷面 rank 而非絕對下月報酬、(ii) 用 12 個 OHLCV-only 特徵而非他們 ~94 個公司特徵 + 8 個 macro、(iii) 強調 post-cost 而非該文 OOS R² 視角。對照 GKX 走「絕對報酬迴歸」研究是 Repo 5 (`ml-return-forecast`) 的主題。
- López de Prado, M. (2018). *Advances in financial machine learning*.
  Wiley. 第 7 章主張金融 cross-validation 應使用 purging + embargo（推薦 CPCV）。本 repo 用的是普通 expanding-window 年度 walk-forward **無 purging** —— 在 21 日 target 視窗 + 年度重訓的設定下尚可（purge 帶來的增益相對於 fold-to-fold IC noise 可忽略），但這是有意偏離書中建議的選擇，production 應該重新檢視。

**特徵歸因**
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS 2017*. SHAP — notebook 03 使用。

**因子衰減與成本**
- Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their
  trading costs. *Review of Financial Studies*, 29(1), 104–147.
  [doi:10.1093/rfs/hhv063](https://doi.org/10.1093/rfs/hhv063) — 設定「gross vs net Sharpe」這個 notebook 04 答的問題。
- McLean, R. D., & Pontiff, J. (2016). Does academic research destroy
  stock return predictability? *Journal of Finance*, 71(1), 5–32.
  [doi:10.1111/jofi.12365](https://doi.org/10.1111/jofi.12365) —
  我們 EDA 的「動量 post-2015 衰減」是這篇 factor-crowding thesis 的具體案例。

**低波異象**
- Baker, M., Bradley, B., & Wurgler, J. (2011). Benchmarks as limits to
  arbitrage: Understanding the low-volatility anomaly. *Financial
  Analysts Journal*, 67(1), 40–54. [doi:10.2469/faj.v67.n1.4](https://doi.org/10.2469/faj.v67.n1.4)
  — 本 EDA 發現 2015–2025 美股樣本下異象 *反轉*（高波勝出）；任何嚴肅報告都應以此文為對照框定反轉。
