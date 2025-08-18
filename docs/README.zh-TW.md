# ml-cross-sectional

> **Languages**: [English](../README.md) · **繁體中文**

我用 12 個 OHLCV 衍生特徵在 S&P 500 上訓練了兩個 cross-sectional ranker（LightGBM 與 XGBoost），年度 walk-forward 驗證、SHAP 歸因、扣完真實成本的 quintile long-short 回測。設計刻意 vanilla —— 真正想看的是 tree model 能不能比一個用同樣特徵手刻的等權 baseline 多做點什麼。

研究框架：[`qtools`](https://github.com/matthiola0/qtools)。對照研究：[`classic-factors`](https://github.com/matthiola0/classic-factors)。

## Headline 結果

`XGBRanker` 模型 + 12 個價量特徵，2020–2024 OOS 在 S&P 500 quintile L/S 上達 **15.4% 年化、Net Sharpe 0.87、MDD −24%** —— 已扣 5 bps 單邊成本。對照 baseline 是 EDA 階段挑出 IC-IR 為正的 3 個特徵等權組合，XGB 約是它的 2× Sharpe / 2× 年化報酬。

Caveat：股票池為當前 S&P 500、有存活者偏差；2022 升息 regime 把所有訊號連 baseline 都打殘。

## 研究問題

我想知道：gradient-boosted ranker 能不能 *系統性* 擊敗一個用同樣特徵手刻的等權 baseline —— 扣完成本、跨 regime 都要成立。但更有意思的問題其實是 *為什麼* 它贏。SHAP 把這個 gap 拆得清楚：tree 在這套特徵上做了線性模型結構上做不到的事，而不只是用更多參數擬合同樣訊號。

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

基本面特徵（P/B、P/E、ROE）刻意 out-of-scope：`qtools` 還沒 fundamentals loader，本研究限於 OHLCV 可推導的訊號。

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

12-1 動量這個發現獨立重現 `classic-factors` 的結論。

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

兩個值得注意的反直覺發現，對照 *ML 交易成本貴* 這個 prior：

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

![net equity curve](../reports/figures/04_net_equity.png)

### 跨股票池 robustness — [notebook 05](../notebooks/05_robustness_tw_btc.ipynb)

相同 12 個特徵、相同 XGB ranker，重訓於台灣 0050（50 檔）與 hard-coded 20 檔 USDT pairs（Binance，2018–2025、OOS 2022–2024）。

| 股票池 | XGB IC-IR | XGB Net Sharpe | `naive_ew` IC-IR | `naive_ew` Net Sharpe |
|---|---|---|---|---|
| 美股（502 檔，2020–2024）| +0.20 | +0.87 | +0.11 | +0.43 |
| 台股 0050（50 檔，2020–2024）| **+0.30** | +0.71 | +0.06 | +0.28 |
| BTC-uni（20 檔，2022–2024）| +0.18 | +0.74 | **−0.39** | **−0.22** |

關鍵觀察是 BTC 上的 `naive_ew` 失敗：3 個手刻特徵（小 ADV long、高波 long、短期反轉 long）在加密的符號 *是反的* —— 小幣 underperform、低波勝出、動量勝過反轉。XGB ranker 訓練時自己學到符號翻轉，仍交出正 IC-IR。**股票池大小（20 vs 500）不重要，重要的是底層市場的符號結構**。

次要發現：台股 IC-IR（+0.30）是 3 個股票池 **最強**，46 bps 來回 `TW_EQUITY` 成本（vs 美股單邊 5 bps）把 Sharpe 從 gross 0.90 砍到 net 0.71 —— 美股砍 6%、台股砍 ~21%。成本明顯但沒有致命：訊號真實、月頻 quintile L/S 在台股是可交易的，只是 cost margin 比美股緊。

### Sector / beta 中性化 — [notebook 06](../notebooks/06_neutralized_backtest.ipynb)

面試最容易攻擊的點 —— 你的 0.87 Sharpe 是不是只是模型從特徵裡吸走的 sector / beta 漂移？這個 notebook 直接答這個問題。每個 cross-section（date × model）把 raw score 對 `[intercept, GICS sector dummies (10), 252d β vs SPY]` 跑 OLS，殘差當中性化分數，**用同一條 04 的回測引擎**重跑（同股票池、同視窗、同 5 bps 單邊成本），只換訊號。

| 模型 | Raw Sharpe | Neutral Sharpe | Raw 年化 | Neutral 年化 | 存活率 | Raw MDD | Neutral MDD |
|---|---|---|---|---|---|---|---|
| `xgb_ranker` | 0.87 | 0.76 | 15.4% | 6.4% | **41%** | −22.7% | **−13.0%** |
| `lgbm_ranker` | 0.70 | **0.85** | 13.6% | 7.4% | **55%** | −25.4% | **−7.9%** |
| `naive_ew` | 0.46 | 0.39 | 7.3% | 3.7% | 50% | −32.8% | −19.6% |
| `linear_lasso` | 0.36 | 0.22 | 5.7% | 1.8% | 32% | −31.1% | −12.8% |
| `linear_ridge` | 0.35 | 0.17 | 5.4% | 1.3% | **23%** | −31.1% | −13.5% |

三個值得指出的讀法：

1. **LGBM 中性化後 Sharpe 反而上升（0.70 → 0.85）**，五個模型裡變最強。看似反直覺，機制其實單純：raw LGBM 在 2022 帶著 sector / beta 漂移正在 *扣分*，剝掉就消掉那年負 Sharpe；per-year 顯示 2022 從 −0.51 翻到 +0.25，**MDD 從 −25% 縮到 −8%** 是同一效應。

2. **XGB 年化報酬腰斬（15.4% → 6.4%）但 Sharpe 只跌 13%**（0.87 → 0.76）。波動同步下降，比值守住。這是教科書級的誠實版本：alpha 是真的，只是比表面數字小 —— 殘下的 6.4% 是選股、消失的 9 pts 是 sector / 高 β 偏倚。

3. **線性 baseline 70–80% alpha 蒸發**：Ridge 存活 23%、Lasso 32%。基本上都是因子曝險不是選股 —— 確認 §EDA 的讀法：regularised linear + 12 特徵集幾乎完全被 small-ADV + high-vol 軸主導，而這條軸跟 sector / beta 高度相關。

中性化後 per-year net Sharpe（對照上面 raw 表）：

| 年份 | XGB | LGBM | naive | lasso | ridge |
|---|---|---|---|---|---|
| 2020 | 1.19 | 1.05 | 0.83 | 0.73 | 0.70 |
| 2021 | −0.01 | 0.77 | 0.24 | 0.42 | 0.35 |
| 2022 | −0.70 | **0.25** | −0.32 | −0.54 | −0.59 |
| 2023 | **2.46** | 1.35 | 1.13 | 0.93 | 0.89 |
| 2024 | 1.12 | 1.01 | 0.14 | −0.80 | −0.91 |

XGB 2023 仍守在 2.46 —— AI rally 的 edge **不是** 純 IT 板塊載荷；但線性模型 2024 從正翻到大負，告訴我它們的 *edge* 是搭板塊趨勢，而非在板塊內挑對股票。

Notebook 沒做的事：真正 PM 會解 sector / beta 約束最佳化，而非殘差化後再平衡。約束法可以保留更多 alpha（任何符合約束的部位都可以拿）；殘差化是保守版本（任何與 sector / beta *相關* 的 alpha 都被剝掉，即使相關是巧合）。存活者偏差跟 point-in-time 成分股的 caveat 同樣適用。

![raw vs neutral equity](../reports/figures/06_raw_vs_neutral_equity.png)

## 失效模式

1. **2022 是所有模型共同的負 Sharpe regime**。升息 drawdown 反向打小型股 / 高波 exposure（5 個訊號共同 load 的主軸），特徵集裡沒有任何宏觀環境感知。對照 `ml-return-forecast` 正是針對此設計，加 macro 特徵（VIX、殖利率、信用利差）。

2. **XGB 在 2021 輸給線性 baseline**：Net Sharpe 0.38 vs 0.90。在平滑延續 regime，COVID 期間訓練的 tree splits 顯然對 2020 反轉 over-fit、漏掉 2021 延續。一個 fold-weighted XGB + Lasso ensemble 應該比單獨 XGB robust。

3. **線性 baseline 輸給 `naive_ew`**：IC-IR 0.07 vs 0.11。12 個特徵裡 4 個 IC 符號是反的（`rsi_14`、`ret_21d`、`macd_hist`、`ret_63d`），tree 可以乾脆不分裂，但 regularised linear regression 必須給係數。Lasso α = 1 × 10⁻⁴ 太弱沒歸零。這是可修的失敗 —— 線性 baseline 應預先用 IC-IR > 0 的子集，已記在 SDD 跟進。

## 限制

- **存活者偏差**：股票池為當前 S&P 500，2020–2024 期間被剔除的不可見。Net Sharpe 估計被高估 10–20%（量級而非實測）。
- **Sector / beta 中性化**：notebook 06 已處理（見上）。誠實答案：tree 模型 41–55% edge 存活、線性 23–32% 存活、MDD 砍半。原本 04 的數字現在要跟中性化表一起讀，不能單看。
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
│   ├── download_sector_beta.py          # Wikipedia GICS + 252d β vs SPY → data/processed/sector_beta.parquet
│   └── build_features.py                # → data/processed/features_*.parquet
├── notebooks/
│   ├── 01_feature_eda.ipynb
│   ├── 02_training_walkforward.ipynb
│   ├── 03_shap_analysis.ipynb
│   ├── 04_backtest.ipynb
│   ├── 05_robustness_tw_btc.ipynb
│   └── 06_neutralized_backtest.ipynb
└── reports/
    ├── figures/                # 入 git；README + notebook 引用
    └── predictions/            # OOS score parquet 給 03 / 04 用
```

## Notebook 導覽

- [`01_feature_eda.ipynb`](../notebooks/01_feature_eda.ipynb) — 12 個 feature 的單因子 Spearman IC 與 IC-IR、重複 feature 剪枝（`low_vol_60` 與 `vol_60d` ρ = −0.999）、post-2015 12-1 動量衰減的獨立重現。
- [`02_training_walkforward.ipynb`](../notebooks/02_training_walkforward.ipynb) — 5 個模型的年度 walk-forward 訓練、pooled OOS IC 表、逐年 IC 矩陣 —— 2022 年 *只有 `naive_ew` 還正* 的 regime 在這裡看到。
- [`03_shap_analysis.ipynb`](../notebooks/03_shap_analysis.ipynb) — 2024 OOS 分層抽樣跑 TreeExplainer。重點發現：`ret_126d` 單因子 IC-IR 只有 −0.03，但 mean |SHAP| 排第三 —— 樹模型條件性使用 feature 勝過 linear 的教科書案例。
- [`04_backtest.ipynb`](../notebooks/04_backtest.ipynb) — Quintile long-short、月頻 rebalance、US_EQUITY 成本。淨 Sharpe 表、逐年拆解、equity curve。標題的 XGB 0.87 Sharpe 在這。
- [`05_robustness_tw_btc.ipynb`](../notebooks/05_robustness_tw_btc.ipynb) — 同 XGB ranker 重訓在台股 0050 與 20 檔 BTC 池。重點是 `naive_ew` 在 BTC 翻負而 XGB 仍 +0.74 Sharpe —— 宇宙大小不如手做特徵的符號是否對得上市場結構重要。
- [`06_neutralized_backtest.ipynb`](../notebooks/06_neutralized_backtest.ipynb) — Sector + 252d β 殘差化後跑 *同一份* backtest。Survival rate、MDD 變化、以及 LGBM 0.70 → 0.85 的反直覺結果。

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

# 2b. （只有 notebook 06 需要）GICS sectors + 252d β vs SPY
python scripts/download_sector_beta.py         # → data/processed/sector_beta.parquet

# 3. 重執行 notebooks（walk-forward 訓練 ~4 分鐘；其餘 < 1 分鐘）
python -m ipykernel install --user --name ml-cross-sectional
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=ml-cross-sectional \
    --inplace notebooks/*.ipynb
```

**本地開發**：若有 [`qtools`](https://github.com/matthiola0/qtools) 本地 clone 想邊改邊用而不 push，安裝後執行 `pip install -e ../qtools` 覆寫 git-installed 版本為 editable 本地版本。

## 參考文獻

**橫斷面 ML in 資產定價**
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *Review of Financial Studies*, 33(5), 2223–2273.
  [doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009) — 線性 / tree / 神經網路在月頻美股報酬預測的 benchmark 比較。本 repo **不是** 直接複製：(i) 用 `LGBMRanker` / `XGBRanker` 預測橫斷面 rank 而非絕對下月報酬、(ii) 用 12 個 OHLCV-only 特徵而非他們 ~94 個公司特徵 + 8 個 macro、(iii) 強調 post-cost 而非該文 OOS R² 視角。對照 GKX 走絕對報酬迴歸的研究是 `ml-return-forecast` 的主題。
- López de Prado, M. (2018). *Advances in financial machine learning*.
  Wiley. 第 7 章主張金融 cross-validation 應使用 purging + embargo（推薦 CPCV）。本 repo 用的是普通 expanding-window 年度 walk-forward **無 purging** —— 在 21 日 target 視窗 + 年度重訓的設定下尚可（purge 帶來的增益相對於 fold-to-fold IC noise 可忽略），但這是有意偏離書中建議的選擇，production 應該重新檢視。

**特徵歸因**
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS 2017*. SHAP — notebook 03 使用。

**因子衰減與成本**
- Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their
  trading costs. *Review of Financial Studies*, 29(1), 104–147.
  [doi:10.1093/rfs/hhv063](https://doi.org/10.1093/rfs/hhv063) — 設定 *gross vs net Sharpe* 這個 notebook 04 答的問題。
- McLean, R. D., & Pontiff, J. (2016). Does academic research destroy
  stock return predictability? *Journal of Finance*, 71(1), 5–32.
  [doi:10.1111/jofi.12365](https://doi.org/10.1111/jofi.12365) —
  EDA 看到的 *動量 post-2015 衰減* 是這篇 factor-crowding thesis 的具體案例。

**低波異象**
- Baker, M., Bradley, B., & Wurgler, J. (2011). Benchmarks as limits to
  arbitrage: Understanding the low-volatility anomaly. *Financial
  Analysts Journal*, 67(1), 40–54. [doi:10.2469/faj.v67.n1.4](https://doi.org/10.2469/faj.v67.n1.4)
  — 本 EDA 發現 2015–2025 美股樣本下異象 *反轉*（高波勝出）；任何嚴肅報告都應以此文為對照框定反轉。
