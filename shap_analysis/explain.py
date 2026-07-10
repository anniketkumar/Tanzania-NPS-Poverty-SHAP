"""Attach SHAP to the fitted model and roll it up into zone-stratified importance.

Three moves, in order:

1. **Explain.** Run `shap.KernelExplainer` on the stacking classifier's
   probability of the "poor" cluster. Model-agnostic KernelExplainer is the right
   tool here: the stack contains a skorch MLP and a logistic meta-learner, so the
   fast TreeExplainer does not apply. It is slow, so by default I summarise the
   background with k-means and explain a stratified per-zone sample of households.

2. **Aggregate one-hot -> feature.** SHAP attributes to the 133/162 one-hot
   columns. For interpretability I sum the |SHAP| of every dummy belonging to the
   same original survey question, giving one importance per real feature (e.g. all
   `toilet_facility_*` columns collapse to `toilet_facility`). This is the
   aggregation approved in Baseline_Replication_Report.md §6.

3. **Stratify by zone.** Attach each household's zone (from its region code via
   `zones.py`), group, and average -> the zone x feature mean|SHAP| matrix that is
   the Week-3 deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from baseline_replication.data import feature_spec as fs

from .config import SHAPConfig
from .model import FittedModel
from .zones import region_code_to_zone, zones_in_use


# Original-feature display order: as they appear in the spec, household_size last.
def _feature_order() -> list[str]:
    order = [f.name for f in fs.ALL_FEATURES]
    for extra in fs.COMPUTED_NUMERIC:          # e.g. household_size
        if extra not in order:
            order.append(extra)
    return order


def build_column_to_feature(columns: list[str]) -> dict[str, str]:
    """Map each design-matrix column back to its origin survey feature.

    Numeric columns equal the feature name (`head_age`). One-hot columns look
    like `f"{feature}_{category}"` (`education_level_11.0`). I match the LONGEST
    feature name that the column equals or starts with (`feature_`), which is
    robust even if one feature name were a prefix of another.
    """
    feature_names = sorted(
        {f.name for f in fs.ALL_FEATURES} | set(fs.COMPUTED_NUMERIC),
        key=len,
        reverse=True,
    )
    mapping: dict[str, str] = {}
    for col in columns:
        for name in feature_names:
            if col == name or col.startswith(name + "_"):
                mapping[col] = name
                break
        else:
            # Shouldn't happen, but keep the column under its own name rather
            # than silently dropping it from the importance totals.
            mapping[col] = col
    return mapping


@dataclass
class SHAPResult:
    """All SHAP artifacts for one wave."""

    shap_onehot: pd.DataFrame        # foreground households x one-hot columns (signed SHAP)
    shap_by_feature: pd.DataFrame    # foreground households x 18 features (|SHAP|), + 'zone'
    zone_importance: pd.DataFrame    # zone x 18 features (mean |SHAP|) -- the deliverable
    expected_value: float            # SHAP base value (mean model output)
    foreground_index: np.ndarray     # household row indices that were explained
    zone_explained_counts: pd.Series # households explained per zone
    zone_total_counts: pd.Series     # total households per zone in the wave


def _stratified_foreground(zone_per_hh: pd.Series, cfg: SHAPConfig) -> np.ndarray:
    """Pick which household rows to explain.

    Full mode: every household with a known zone. Sampled mode: up to
    `per_zone_sample` households per zone, chosen reproducibly from `seed`.
    """
    valid = zone_per_hh.dropna()
    if cfg.full:
        return valid.index.to_numpy()

    rng = np.random.RandomState(cfg.seed)
    picks: list[int] = []
    for zone, idx in valid.groupby(valid).groups.items():
        idx = np.asarray(idx)
        if len(idx) > cfg.per_zone_sample:
            idx = rng.choice(idx, size=cfg.per_zone_sample, replace=False)
        picks.extend(idx.tolist())
    return np.sort(np.asarray(picks))


def run_shap(
    model: FittedModel,
    cfg: SHAPConfig,
    exclude_features: list[str] | None = None,
) -> SHAPResult:
    """Compute per-household SHAP, aggregate to features, stratify by zone.

    Parameters
    ----------
    exclude_features : list[str] | None
        Feature names (original, not one-hot) to drop from the zone-importance
        matrix *after* SHAP computation.  Useful for producing a heatmap without
        geographic features (``region``, ``rural_urban``) to discuss the
        tautology risk.  The raw per-household SHAP values are unaffected.
    """
    X = model.X_imp
    columns = model.feature_columns

    # Household -> zone (NaN for unknown / excluded Zanzibar codes).
    zone_per_hh = model.region.map(
        lambda c: region_code_to_zone(c, include_zanzibar=cfg.include_zanzibar)
    )

    foreground_idx = _stratified_foreground(zone_per_hh, cfg)
    if len(foreground_idx) == 0:
        raise ValueError("No households selected for SHAP — check the zone mapping.")
    X_fg = X.iloc[foreground_idx]

    # ---- 1. SHAP KernelExplainer on P(poor) ------------------------------ #
    # Seed the global NumPy RNG: both shap.kmeans (its internal sklearn KMeans
    # uses the global RNG when no random_state is given) and KernelExplainer's
    # subset sampling draw from it. Without this the zone matrix wobbles by
    # ~0.01 mean|SHAP| run-to-run (rankings stay put, but values aren't
    # bit-reproducible). Seeding makes the whole SHAP run deterministic.
    np.random.seed(cfg.seed)
    background = shap.kmeans(X.to_numpy(), min(cfg.background_size, len(X)))
    clf = model.clf

    def f(data: np.ndarray) -> np.ndarray:
        return clf.predict_proba(data)[:, cfg.explain_class]

    explainer = shap.KernelExplainer(f, background)
    raw_vals = explainer.shap_values(X_fg.to_numpy(), nsamples=cfg.nsamples)
    # For a scalar-output function KernelExplainer returns (n_fg, n_features).
    shap_vals = np.asarray(raw_vals)
    if shap_vals.ndim == 3:            # defensive: some versions add a class axis
        shap_vals = shap_vals[..., cfg.explain_class]

    shap_onehot = pd.DataFrame(shap_vals, index=X_fg.index, columns=columns)

    # ---- 2. Aggregate one-hot |SHAP| -> original features ---------------- #
    col2feat = build_column_to_feature(columns)
    abs_onehot = shap_onehot.abs()
    by_feature = abs_onehot.T.groupby(pd.Series(col2feat)).sum().T
    ordered = [f_ for f_ in _feature_order() if f_ in by_feature.columns]
    by_feature = by_feature[ordered]
    by_feature.insert(0, "zone", zone_per_hh.loc[X_fg.index].to_numpy())

    # ---- 3. Zone-stratified mean importance (the deliverable) ------------ #
    active_zones = [z for z in zones_in_use(cfg.include_zanzibar) if z in set(by_feature["zone"])]

    # Optionally drop geographic features from the importance matrix.
    report_cols = ordered
    if exclude_features:
        report_cols = [f for f in ordered if f not in exclude_features]

    zone_importance = (
        by_feature.groupby("zone")[report_cols].mean()
        .reindex(active_zones)
    )

    # Per-zone household counts (total in wave + explained in this run).
    zone_total_counts = (
        zone_per_hh.dropna().value_counts()
        .reindex(active_zones, fill_value=0)
        .rename("n_total")
    )
    zone_explained_counts = (
        zone_per_hh.loc[foreground_idx].value_counts()
        .reindex(active_zones, fill_value=0)
        .rename("n_explained")
    )

    expected = float(np.asarray(explainer.expected_value).ravel()[0])

    return SHAPResult(
        shap_onehot=shap_onehot,
        shap_by_feature=by_feature,
        zone_importance=zone_importance,
        expected_value=expected,
        foreground_index=foreground_idx,
        zone_explained_counts=zone_explained_counts,
        zone_total_counts=zone_total_counts,
    )
