"""The stacked ensemble, built with scikit-learn's StackingClassifier.

This stands in for the H2O stacked ensemble from the paper. The idea of stacking
is: train several different kinds of model, then train one more model (the
"meta-learner") to combine their predictions. My base models cover the same
families the paper used — a linear model, random forest, gradient boosting,
XGBoost, and a neural net. The neural net is the PyTorch/skorch MLP from nn.py,
wrapped so its inputs get cast to float32.

XGBoost is optional: if it isn't installed I swap in a second gradient-boosting
model and note that I did, so a missing package never crashes the whole run.
"""

from __future__ import annotations

import warnings

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ..config import NNConfig, StackingConfig
from .nn import FloatCastWrapper, build_nn_classifier

# XGBoost is nice to have but not required, so import it defensively.
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_XGB = False


def _make_base_learner(
    name: str,
    *,
    input_dim: int,
    n_classes: int,
    nn_cfg: NNConfig,
    seed: int,
):
    name = name.lower()
    if name == "glm":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "gbm":
        return GradientBoostingClassifier(random_state=seed)
    if name == "drf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == "xgb":
        if _HAS_XGB:
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
        warnings.warn(
            "xgboost not installed; substituting a GradientBoosting estimator "
            "for the 'xgb' base learner.",
            RuntimeWarning,
        )
        return GradientBoostingClassifier(random_state=seed + 1)
    if name == "nn":
        net = build_nn_classifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=nn_cfg.hidden_dims,
            dropout=nn_cfg.dropout,
            max_epochs=nn_cfg.max_epochs,
            lr=nn_cfg.lr,
            batch_size=nn_cfg.batch_size,
            weight_decay=nn_cfg.weight_decay,
            device=nn_cfg.device,
            seed=seed,
        )
        # The rest of the pipeline passes float64 arrays around, but torch wants
        # float32, so put a cast step in front of the net.
        return Pipeline(
            [
                ("cast", FunctionTransformer(FloatCastWrapper(), validate=False)),
                ("net", net),
            ]
        )
    raise ValueError(f"Unknown base learner {name!r}.")


def _make_meta_learner(name: str, seed: int):
    name = name.lower()
    if name == "glm":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "gbm":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown meta-learner {name!r}.")


def build_stacking_classifier(
    cfg: StackingConfig,
    nn_cfg: NNConfig,
    *,
    input_dim: int,
    n_classes: int,
    seed: int,
) -> StackingClassifier:
    """Build the base learners and meta-learner and wire them into a stacker."""
    estimators = [
        (name, _make_base_learner(
            name, input_dim=input_dim, n_classes=n_classes, nn_cfg=nn_cfg, seed=seed
        ))
        for name in cfg.base_learners
    ]
    final_estimator = _make_meta_learner(cfg.meta_learner, seed)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cfg.cv,
        stack_method=cfg.stack_method,
        passthrough=cfg.passthrough,
        n_jobs=cfg.n_jobs,
    )


def xgboost_available() -> bool:
    return _HAS_XGB
