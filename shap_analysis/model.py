"""Fit the poverty-clustering model once, on the full wave, ready for SHAP.

The baseline's experiment loop repeats a train/test split over 20 seeds to
*measure* the classifier. SHAP is different: I need ONE fitted model to explain,
and I want it fit on every household so the explanation covers the whole survey.
So this module runs the same stages as `baseline_replication.pipeline.experiment`
(KNN impute -> PCA -> K-Means -> Stacked Ensemble) but on the full data with a
single seed, and hands back the pieces SHAP needs.

Crucially, the classifier is trained on the RAW imputed feature matrix (the
baseline's `classifier_input="raw"` default), NOT on the PCA coordinates. That is
the "no-PCA model" the Week-3 task calls for: SHAP values then map to real survey
questions (floor material, toilet type, ...) rather than abstract components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from baseline_replication.data import build_design_matrix, load_wave_features
from baseline_replication.pipeline.clustering import fit_clusterer
from baseline_replication.pipeline.imputation import build_imputer
from baseline_replication.pipeline.reduction import build_reducer
from baseline_replication.pipeline.stacking import build_stacking_classifier

from .config import SHAPConfig


@dataclass
class FittedModel:
    """Everything downstream SHAP code needs from a single fitted pipeline."""

    clf: object                 # the fitted StackingClassifier (trained on raw features)
    X_imp: pd.DataFrame         # imputed raw feature matrix (the classifier's input space)
    labels: np.ndarray          # K-Means cluster label per household (0/1)
    region: pd.Series           # raw region code per household, row-aligned with X_imp
    feature_columns: list[str]  # design-matrix column names (one-hot + numeric)
    cluster_info: dict          # k, label_map, train_label_counts from the clusterer


def fit_full_model(cfg: SHAPConfig) -> FittedModel:
    """Build a wave, run impute -> PCA -> K-Means -> stacker on ALL of it."""
    base = cfg.synced_baseline()
    seed = cfg.seed

    # ---- assemble the wave (raw features, one row per household) --------- #
    raw = load_wave_features(
        base.data.converted_data_dir, base.data.wave, base.data.head_of_household_only
    )
    if "region" not in raw.columns:
        raise KeyError(
            "The loader did not return a 'region' column; zone stratification "
            "needs it. Check baseline_replication.data.feature_spec.GEOGRAPHIC."
        )
    region = raw["region"].reset_index(drop=True)

    design = build_design_matrix(raw)
    if base.data.dropna_before_impute:
        keep = design.dropna().index
        design = design.loc[keep]
        region = region.loc[keep].reset_index(drop=True)
    feature_columns = list(design.columns)

    # ---- KNN imputation (fit on the full matrix) ------------------------- #
    X = design.to_numpy(dtype=float)
    if base.impute.enabled:
        imputer = build_imputer(base.impute)
        X_imp = imputer.fit_transform(X)
    else:
        X_imp = X
    X_imp_df = pd.DataFrame(X_imp, columns=feature_columns).reset_index(drop=True)

    # ---- PCA (only to build the K-Means label) --------------------------- #
    # PCA is used ONLY to manufacture the poverty label, exactly as in the paper.
    # The classifier and SHAP never see these components.
    reducer = build_reducer(base.reduction, seed)
    X_red = reducer.fit_transform(X_imp)

    # ---- K-Means -> poverty label ---------------------------------------- #
    clusterer, cluster_info = fit_clusterer(X_red, base.cluster, seed)
    labels = clusterer.predict(X_red)

    # ---- Stacked ensemble on the RAW imputed features -------------------- #
    n_classes = len(np.unique(labels))
    clf = build_stacking_classifier(
        base.stacking,
        base.nn,
        input_dim=X_imp.shape[1],
        n_classes=n_classes,
        seed=seed,
    )
    clf.fit(X_imp, labels)

    return FittedModel(
        clf=clf,
        X_imp=X_imp_df,
        labels=labels,
        region=region,
        feature_columns=feature_columns,
        cluster_info=cluster_info,
    )
