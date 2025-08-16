"""Walk-forward mask correctness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlcs.validation import walk_forward_years


def test_walk_forward_masks_are_disjoint_and_causal():
    dates = pd.Series(pd.bdate_range("2018-01-01", "2022-12-31"))
    folds = list(walk_forward_years(dates, first_oos_year=2020, last_oos_year=2022))

    assert [y for y, _, _ in folds] == [2020, 2021, 2022]

    years = dates.dt.year.values
    for oos_year, train_mask, test_mask in folds:
        # strictly disjoint
        assert not np.any(train_mask & test_mask)
        # causal: every training date strictly precedes the OOS year
        assert years[train_mask].max() < oos_year
        # test_mask is exactly the OOS year
        assert set(years[test_mask].tolist()) == {oos_year}


def test_walk_forward_skips_years_without_both_sides():
    dates = pd.Series(pd.bdate_range("2020-01-01", "2021-12-31"))
    # 2020 has no train (nothing before 2020); 2022+ has no test data.
    folds = list(walk_forward_years(dates, first_oos_year=2020, last_oos_year=2024))
    assert [y for y, _, _ in folds] == [2021]
