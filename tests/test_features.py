"""Correctness checks for the feature panel and forward target."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlcs.features import add_forward_target, build_feature_panel


@pytest.fixture
def panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=400)
    symbols = [f"S{i:02d}" for i in range(8)]
    # geometric random walk per symbol so returns / vol behave reasonably
    rets = rng.normal(0.0004, 0.015, size=(len(dates), len(symbols)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=symbols)
    volume = pd.DataFrame(
        rng.integers(1_000_000, 5_000_000, size=close.shape), index=dates, columns=symbols
    ).astype("float64")
    return close, volume


def test_feature_panel_has_expected_columns(panel):
    close, volume = panel
    features = build_feature_panel(close, volume)
    expected = {
        "mom_12_1", "reversal_1w", "low_vol_60", "size_adv_60",
        "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "vol_20d", "vol_60d", "rsi_14", "macd_hist", "volume_z_60",
    }
    assert expected.issubset(features.columns)
    assert {"date", "symbol"}.issubset(features.columns)


def test_forward_target_aligns_with_future_return(panel):
    close, volume = panel
    features = build_feature_panel(close, volume)
    horizon = 21
    features = add_forward_target(features, close, horizon=horizon)

    # pick a mid-panel (date, symbol) with both sides defined
    row = features.dropna(subset=[f"fwd_ret_{horizon}d"]).iloc[500]
    d, s = row["date"], row["symbol"]
    expected = close.shift(-horizon).loc[d, s] / close.loc[d, s] - 1
    assert row[f"fwd_ret_{horizon}d"] == pytest.approx(expected, rel=1e-9)


def test_forward_rank_is_pct_within_date(panel):
    close, volume = panel
    features = build_feature_panel(close, volume)
    features = add_forward_target(features, close, horizon=21)

    # pct=True ranks lie in (0, 1]; per date, max rank must equal 1 when any
    # non-NaN forward returns exist (ties broken by average rank).
    sample = features.dropna(subset=["fwd_rank_21d"])
    max_by_date = sample.groupby("date")["fwd_rank_21d"].max()
    assert (max_by_date <= 1.0 + 1e-9).all()
    assert (max_by_date >= 0.5).all()  # at least two symbols per date in fixture
