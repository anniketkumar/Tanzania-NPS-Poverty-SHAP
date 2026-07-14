"""The loop that runs the whole pipeline once per seed and ties everything together.

For each random seed:
    1. split into train/test (30% test)
    2. KNN imputation      -- fit on train, only transform test  ("KNN")
    3. PCA                 -- fit on train, only transform test  ("PCA")
    4. K-Means on the reduced train set, then predict labels for both  ("K-Means")
    5. train the StackingClassifier on the chosen feature space  ("Stacked Ensemble")
    6. score it on the held-out test rows

Fitting each transform on the training rows only (and just transforming the test
rows) is the part that keeps the test set out of the fitting. I then average the
metrics across all seeds (mean +/- std) and dump everything to an Excel file.

Important caveat about the default setup: when there's no independent label, the
target IS the K-Means label, and K-Means built that label from these same
features. So the classifier is essentially being asked to re-trace a boundary
that's already a deterministic function of the inputs — see run_single_seed for
the full note. That's why the accuracy comes out near-perfect; it's not the same
thing as predicting real-world poverty.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from ..config import Config
from . import metrics as M
from .clustering import fit_clusterer
from .imputation import build_imputer
from .reduction import build_reducer
from .stacking import build_stacking_classifier, xgboost_available


def _split(X, y_ext, test_size, seed):
    """Split into train/test. If I have a real label, stratify on it so both
    halves keep the same class balance; otherwise just do a plain random split
    (I don't have the K-Means label yet at this point)."""
    stratify = y_ext if y_ext is not None else None
    if y_ext is None:
        X_tr, X_te = train_test_split(
            X, test_size=test_size, random_state=seed, shuffle=True
        )
        return X_tr, X_te, None, None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_ext, test_size=test_size, random_state=seed, stratify=stratify
    )
    return X_tr, X_te, y_tr, y_te


def run_single_seed(design: pd.DataFrame, y_ext, cfg: Config, seed: int) -> dict:
    """Run the full pipeline once for one seed and return its metrics + artifacts."""
    X = design.to_numpy(dtype=float)
    X_tr, X_te, y_tr_ext, y_te_ext = _split(X, y_ext, cfg.experiment.test_size, seed)

    # ---- 2. KNN imputation --------------------------------------------- #
    # Learn the neighbour structure from the training rows, then use it to fill
    # gaps in both sets. The test rows never influence what gets learned.
    if cfg.impute.enabled:
        imputer = build_imputer(cfg.impute)
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)
    else:
        X_tr_imp, X_te_imp = X_tr, X_te

    # ---- 3. PCA (with its scaler) -------------------------------------- #
    # Same pattern: fit the scaler + PCA on train only, then project test.
    reducer = build_reducer(cfg.reduction, seed)
    X_tr_red = reducer.fit_transform(X_tr_imp)
    X_te_red = reducer.transform(X_te_imp)

    # ---- 4. K-Means -> poverty label ----------------------------------- #
    # Fit K-Means on the reduced TRAIN data, then assign every row (train and
    # test) to its nearest centroid. Fitting on train only means the test
    # assignments come from centroids the test data didn't help place.
    clusterer, cluster_info = fit_clusterer(X_tr_red, cfg.cluster, seed)
    y_tr_clust = clusterer.predict(X_tr_red)
    y_te_clust = clusterer.predict(X_te_red)

    # ---- Cluster-validity indices (on the reduced TRAIN space) ---------- #
    # These measure how separated the K-Means clusters are, independent of the
    # classifier. Computed on the same space K-Means used (train, reduced).
    # Higher silhouette / Calinski-Harabasz = better; lower Davies-Bouldin = better.
    cluster_quality = {"silhouette": float("nan"),
                       "calinski_harabasz": float("nan"),
                       "davies_bouldin": float("nan")}
    if len(np.unique(y_tr_clust)) > 1:
        cluster_quality = {
            "silhouette": float(silhouette_score(X_tr_red, y_tr_clust)),
            "calinski_harabasz": float(calinski_harabasz_score(X_tr_red, y_tr_clust)),
            "davies_bouldin": float(davies_bouldin_score(X_tr_red, y_tr_clust)),
        }

    # Pick the target. If I supplied a real independent label, use it. Otherwise
    # fall back to the K-Means label — and note that this is the circular case:
    # the label is just a geometric partition of the features, so a flexible
    # classifier can nearly reconstruct it. High accuracy here measures
    # "can the ensemble redraw the K-Means boundary", not real poverty prediction.
    if y_ext is not None:
        y_tr = np.asarray(y_tr_ext)
        y_te = np.asarray(y_te_ext)
        target_name = cfg.experiment.true_label_col
    else:
        y_tr, y_te = y_tr_clust, y_te_clust
        target_name = "cluster_label"

    # ---- 5. Pick the feature space the classifier sees ----------------- #
    # "reduced" hands it the same 2 PCA coords K-Means used (label becomes almost
    # trivial to recover); "raw" uses the full imputed feature set, which is a
    # bit more honest but still ultimately a function of the same inputs.
    if cfg.stacking.classifier_input == "reduced":
        X_clf_tr, X_clf_te = X_tr_red, X_te_red
    else:  # "raw"
        X_clf_tr, X_clf_te = X_tr_imp, X_te_imp

    n_classes = len(np.unique(y_tr))
    clf = build_stacking_classifier(
        cfg.stacking,
        cfg.nn,
        input_dim=X_clf_tr.shape[1],
        n_classes=n_classes,
        seed=seed,
    )
    clf.fit(X_clf_tr, y_tr)

    y_pred = clf.predict(X_clf_te)
    proba = clf.predict_proba(X_clf_te)
    classes = list(clf.classes_)

    metrics = M.compute_metrics_all(y_te, y_pred, proba, classes)

    # Train-set accuracy too, so we can report the train-vs-test gap (an
    # over-fitting check). `accuracy` in `metrics` is the test-set figure.
    from sklearn.metrics import accuracy_score
    train_accuracy = float(accuracy_score(y_tr, clf.predict(X_clf_tr)))
    cluster_quality["train_accuracy"] = train_accuracy
    cluster_quality["train_test_gap"] = train_accuracy - float(metrics.get("accuracy", float("nan")))

    return {
        "seed": seed,
        "target": target_name,
        "k": cluster_info["k"],
        "metrics": metrics,
        "cluster_quality": cluster_quality,
        "confusion": M.confusion(y_te, y_pred),
        "per_class": M.per_class_metrics(y_te, y_pred),
        "cluster_info": cluster_info,
    }


def run_experiment(design: pd.DataFrame, y_ext, cfg: Config) -> dict:
    """Run the pipeline for every seed and collect the results into DataFrames."""
    cfg.experiment.output_dir.mkdir(parents=True, exist_ok=True)

    rows, confusions, per_class_frames = [], [], []
    for i, seed in enumerate(cfg.experiment.seeds, start=1):
        print(f"[run {i}/{len(cfg.experiment.seeds)}] seed={seed} ...", flush=True)
        res = run_single_seed(design, y_ext, cfg, seed)
        rows.append({"run": i, "seed": seed, "target": res["target"],
                     "k": res["k"], **res["metrics"], **res["cluster_quality"]})
        confusions.append({"run": i, "seed": seed, "confusion": res["confusion"]})
        per_class_frames.append(res["per_class"].assign(run=i, seed=seed))

    summary = pd.DataFrame(rows)
    metric_cols = [c for c in summary.columns if c not in ("run", "seed", "target", "k")]
    aggregate = (
        summary[metric_cols]
        .agg(["mean", "std"])
        .T.rename(columns={"mean": "mean", "std": "std"})
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    confusion_df = pd.DataFrame(confusions)
    per_class_df = pd.concat(per_class_frames, ignore_index=True)

    results = {
        "summary": summary,
        "aggregate": aggregate,
        "confusion": confusion_df,
        "per_class": per_class_df,
    }
    _export_excel(results, cfg)
    if cfg.experiment.make_plots:
        _make_plots(summary, cfg)
    return results


def _export_excel(results: dict, cfg: Config) -> None:
    out = cfg.experiment.output_dir / cfg.experiment.excel_out
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        results["summary"].to_excel(writer, sheet_name="summary", index=False)
        results["aggregate"].to_excel(writer, sheet_name="aggregate", index=False)
        results["confusion"].to_excel(writer, sheet_name="confusion_matrices", index=False)
        results["per_class"].to_excel(writer, sheet_name="classwise_metrics", index=False)
        pd.DataFrame(
            [{"setting": k, "value": str(v)} for k, v in _flat_config(cfg).items()]
        ).to_excel(writer, sheet_name="config", index=False)
    print(f"\nSaved results -> {out}")


def _flat_config(cfg: Config) -> dict:
    flat = {}
    for section, sub in asdict(cfg).items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                flat[f"{section}.{k}"] = v
        else:
            flat[section] = sub
    flat["xgboost_available"] = xgboost_available()
    return flat


def _make_plots(summary: pd.DataFrame, cfg: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ("accuracy", "f1", "auc_macro"):
        if metric in summary.columns:
            ax.plot(summary["run"], summary[metric], marker="o", label=metric)
    ax.set_title("Stacked-ensemble metrics across seeds")
    ax.set_xlabel("Run")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = cfg.experiment.output_dir / f"learning_curve_wave{cfg.data.wave}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved plot    -> {path}")
