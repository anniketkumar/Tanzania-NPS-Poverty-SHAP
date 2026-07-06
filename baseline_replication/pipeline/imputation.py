"""KNN imputation — the 'KNN' step.

Lots of cells in this survey are missing (roughly 9-12% for most features, and
much more for the electricity-source column). Instead of dropping those rows, I
fill each gap using the values of the nearest complete rows. This is just a thin
factory returning the imputer; the experiment loop is what fits it on the
training rows and only transforms the test rows, so the test data stays out of
the fit.
"""

from __future__ import annotations

from sklearn.impute import KNNImputer

from ..config import ImputeConfig


def build_imputer(cfg: ImputeConfig) -> KNNImputer:
    return KNNImputer(n_neighbors=cfg.n_neighbors, weights=cfg.weights)
