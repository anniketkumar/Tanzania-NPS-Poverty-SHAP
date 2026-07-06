"""One-off diagnostic: is the ~99% accuracy real, or an artifact of the
K-Means-label circularity? Not part of the pipeline; safe to delete.

It runs four checks on the real wave-4 data:
  A. Proper pipeline (fit imputer/PCA/K-Means on TRAIN only): train vs test gap.
  B. Same, but the classifier sees the 2 PCA coords K-Means used ("reduced").
  C. Proper stratified 5-fold CV, regenerating the K-Means label inside each
     fold's training data, to see whether ~99% survives out-of-sample.
  D. Control: replace the target with a RANDOM label unrelated to the features
     and rerun. If accuracy collapses to chance, the ~99% in A/C was coming
     entirely from the label being a function of the features.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from aniket_code.config import default_config
from aniket_code.data import build_design_matrix, load_wave_features
from aniket_code.pipeline.imputation import build_imputer
from aniket_code.pipeline.reduction import build_reducer
from aniket_code.pipeline.clustering import fit_clusterer
from aniket_code.pipeline.stacking import build_stacking_classifier


def _prep(cfg, X_tr, X_te, seed):
    """Impute + reduce with everything fit on train only. Returns imputed and
    reduced versions of both splits."""
    imp = build_imputer(cfg.impute)
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)
    red = build_reducer(cfg.reduction, seed)
    X_tr_r = red.fit_transform(X_tr_i)
    X_te_r = red.transform(X_te_i)
    return X_tr_i, X_te_i, X_tr_r, X_te_r


def _fit_clf(cfg, X_tr, y_tr, seed):
    clf = build_stacking_classifier(
        cfg.stacking, cfg.nn,
        input_dim=X_tr.shape[1], n_classes=len(np.unique(y_tr)), seed=seed,
    )
    clf.fit(X_tr, y_tr)
    return clf


def main():
    cfg = default_config()
    # Trim the NN so this probe finishes quickly; still the full 5-learner stack.
    cfg.nn.max_epochs = 15

    raw = load_wave_features(cfg.data.converted_data_dir, 4, True)
    design = build_design_matrix(raw)
    X = design.to_numpy(dtype=float)
    print(f"wave 4 design matrix: {X.shape[0]} households x {X.shape[1]} cols\n")

    seeds = [42, 43, 44]

    # ---- A & B: train vs test gap, raw-input and reduced-input ------------ #
    print("=== A/B: train vs test accuracy (target = K-Means label) ===")
    for seed in seeds:
        X_tr, X_te = train_test_split(X, test_size=0.30, random_state=seed, shuffle=True)
        X_tr_i, X_te_i, X_tr_r, X_te_r = _prep(cfg, X_tr, X_te, seed)

        clusterer, _ = fit_clusterer(X_tr_r, cfg.cluster, seed)
        y_tr = clusterer.predict(X_tr_r)
        y_te = clusterer.predict(X_te_r)

        # A: classifier on raw imputed features
        clf = _fit_clf(cfg, X_tr_i, y_tr, seed)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr_i))
        te_acc = accuracy_score(y_te, clf.predict(X_te_i))

        # B: classifier on the 2 PCA coords K-Means itself used
        clf_r = _fit_clf(cfg, X_tr_r, y_tr, seed)
        te_acc_r = accuracy_score(y_te, clf_r.predict(X_te_r))

        print(f"  seed {seed}: RAW train={tr_acc:.4f} test={te_acc:.4f} "
              f"gap={tr_acc - te_acc:+.4f} | REDUCED test={te_acc_r:.4f}")

    # ---- C: proper stratified 5-fold CV, label regenerated per fold ------- #
    print("\n=== C: 5-fold CV (K-Means label refit inside each fold) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Need a label to stratify the folds; use a one-shot full-data clustering
    # ONLY for choosing fold membership (not for scoring).
    imp0 = build_imputer(cfg.impute); Xi0 = imp0.fit_transform(X)
    red0 = build_reducer(cfg.reduction, 42); Xr0 = red0.fit_transform(Xi0)
    strat, _ = fit_clusterer(Xr0, cfg.cluster, 42)
    strat_labels = strat.predict(Xr0)

    cv_acc, cv_auc = [], []
    for k, (tr_idx, te_idx) in enumerate(skf.split(X, strat_labels), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        X_tr_i, X_te_i, X_tr_r, X_te_r = _prep(cfg, X_tr, X_te, 42)
        clusterer, _ = fit_clusterer(X_tr_r, cfg.cluster, 42)  # refit on fold train
        y_tr = clusterer.predict(X_tr_r)
        y_te = clusterer.predict(X_te_r)
        clf = _fit_clf(cfg, X_tr_i, y_tr, 42)
        acc = accuracy_score(y_te, clf.predict(X_te_i))
        proba = clf.predict_proba(X_te_i)[:, 1]
        auc = roc_auc_score(y_te, proba)
        cv_acc.append(acc); cv_auc.append(auc)
        print(f"  fold {k}: acc={acc:.4f} auc={auc:.4f}")
    print(f"  CV mean acc={np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f} | "
          f"mean auc={np.mean(cv_auc):.4f}")

    # ---- D: control with a RANDOM label unrelated to features ------------- #
    print("\n=== D: control -- random label independent of features ===")
    rng = np.random.default_rng(0)
    for seed in seeds:
        X_tr, X_te, y_tr_rand, y_te_rand = train_test_split(
            X, rng.integers(0, 2, size=X.shape[0]),
            test_size=0.30, random_state=seed, shuffle=True,
        )
        X_tr_i, X_te_i, _, _ = _prep(cfg, X_tr, X_te, seed)
        clf = _fit_clf(cfg, X_tr_i, y_tr_rand, seed)
        te_acc = accuracy_score(y_te_rand, clf.predict(X_te_i))
        print(f"  seed {seed}: test acc on RANDOM label = {te_acc:.4f} (chance ~0.50)")


if __name__ == "__main__":
    main()