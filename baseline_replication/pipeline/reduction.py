"""Dimensionality reduction that feeds K-Means.

My paper source uses PCA, so PCA is the only real reducer here. I standardize
the features first (mean 0, unit variance) because PCA looks at variance, and
without scaling whichever feature happens to have the largest raw numbers would
dominate the components.

I return an *unfitted* sklearn Pipeline rather than a fitted object on purpose:
the experiment loop fits it on the training rows only and then just transforms
the test rows, which is what keeps the test set from leaking into the fit.

The "none" option is a debugging escape hatch — it skips PCA and hands K-Means
the standardized features directly, so I can see what clustering looks like
without any reduction.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ReductionConfig


def build_reducer(cfg: ReductionConfig, seed: int) -> Pipeline:
    steps = []
    if cfg.standardize:
        steps.append(("scaler", StandardScaler()))

    method = cfg.method.lower()
    if method == "pca":
        steps.append(("reducer", PCA(n_components=cfg.n_components, random_state=seed)))
    elif method == "none":
        pass  # keep the standardized features, no reduction
    else:
        raise ValueError(
            f"Unknown reduction method {cfg.method!r}; expected 'pca' or 'none'."
        )

    if not steps:
        # Nothing to do (no scaling, no reduction): pass the data through untouched.
        steps.append(("scaler", StandardScaler(with_mean=False, with_std=False)))
    return Pipeline(steps)