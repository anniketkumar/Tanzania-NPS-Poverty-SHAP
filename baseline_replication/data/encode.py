"""Turn the raw feature table into an all-numeric matrix the models can use.

Most of my features are categorical codes (region, toilet type, and so on), and
the models need numbers, so I one-hot encode them: each category becomes its own
0/1 column. The numeric features (age, rooms, ...) pass straight through. The
one subtlety is missing values: I deliberately keep them as NaN instead of
dropping the row or treating "missing" as a category, because the KNN imputer in
the next step is what's supposed to fill them in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import feature_spec as fs


def build_design_matrix(
    raw: pd.DataFrame,
    drop_first: bool = False,
) -> pd.DataFrame:
    """Build the all-numeric matrix: categoricals become one-hot columns, and a
    missing category becomes an all-NaN row across its dummies (so the imputer
    can fill it) rather than a row of zeros. Numeric columns pass through.
    """
    cat_names = [c for c in fs.categorical_feature_names() if c in raw.columns]
    num_names = [c for c in fs.numeric_feature_names() if c in raw.columns]

    # Numeric columns: force them to real numbers; anything unparseable becomes
    # NaN, which the imputer will handle later.
    numeric = raw[num_names].apply(pd.to_numeric, errors="coerce")

    # Categorical columns: one-hot each one. get_dummies would normally turn a
    # missing value into an all-zero row, which the model would read as a
    # confident "none of these categories". That's wrong here, so I find those
    # rows and set their dummies back to NaN — that way it reads as "unknown"
    # and the imputer gets a chance to fill it in.
    cat_frames = []
    for col in cat_names:
        s = raw[col].astype("category")
        dummies = pd.get_dummies(s, prefix=col, dummy_na=False, drop_first=drop_first)
        dummies = dummies.astype(float)
        missing_mask = raw[col].isna().to_numpy()
        if missing_mask.any():
            dummies.loc[missing_mask, :] = np.nan
        cat_frames.append(dummies)

    blocks = [numeric] + cat_frames
    design = pd.concat(blocks, axis=1)
    design.index = raw.index
    return design
