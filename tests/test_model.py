"""Rank labels and min-coverage filtering for the LGBM / XGB rankers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlcs.model import _decile_labels


def test_decile_labels_are_integer_bins_per_date():
    rng = np.random.default_rng(1)
    n_dates, n_syms = 10, 40
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_syms)
    y = pd.Series(rng.normal(size=n_dates * n_syms))

    labels = _decile_labels(y, pd.Series(dates), n_bins=10)

    # per date: labels span 0..9 (or a subset thereof), never exceed n_bins-1
    by_date = pd.DataFrame({"d": dates, "lab": labels.values}).dropna()
    for _, grp in by_date.groupby("d"):
        uniq = set(grp["lab"].astype(int).unique())
        assert uniq.issubset(set(range(10)))
        assert len(uniq) == 10  # 40 obs / 10 bins = 4 per bin, no ties


def test_decile_labels_nan_when_insufficient_rows():
    # only 5 rows for a single date, but 10 bins requested → all NaN
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    dates = pd.Series(pd.to_datetime(["2020-01-01"] * 5))
    labels = _decile_labels(y, dates, n_bins=10)
    assert labels.isna().all()
