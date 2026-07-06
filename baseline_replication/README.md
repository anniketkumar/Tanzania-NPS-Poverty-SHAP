# Poverty-clustering baseline (Tanzania NPS)

My replication of the poverty-clustering methodology: build household features
from the survey, fill gaps, reduce, cluster into poor / non-poor, then train a
stacked ensemble to predict that cluster.

```
raw NPS CSVs ─▶ ~28 features ─▶ one-hot ─▶ KNN impute ─▶ PCA ─▶ K-Means ─▶ Stacked Ensemble
                                          └─ "KNN" ──┘  └PCA┘  └K-Means┘   └ scikit-learn ┘
```

This is **baseline replication only** — no new features and no SHAP yet; those
come in the next phase (Week 3).

Two tooling choices, made because I'm rebuilding this in a plain scikit-learn /
PyTorch stack rather than H2O:

| Reference method | What I use here |
|---|---|
| H2O AutoML stacked ensemble | scikit-learn `StackingClassifier` |
| H2O deep-learning model | PyTorch MLP wrapped in **skorch** |

Reduction is **PCA only** — that's what my source uses.

---

## Quick start

```bash
# from the project root (the folder that contains aniket_code/)
pip install -r aniket_code/requirements.txt

# 1) verify the wiring on synthetic data (seconds, no NPS files needed)
python -m aniket_code.smoke_test

# 2) run the real replication on wave 4 (or 5)
python -m aniket_code.run_baseline --wave 4            # all 20 seeds
python -m aniket_code.run_baseline --wave 4 --seeds 3  # quick 3-seed run
```

Results (per-seed metrics, aggregate mean±std, confusion matrices, the config
used) land in `aniket_code/outputs/experimental_summary.xlsx`, plus a
`learning_curve.png`.

**On this machine:** the smoke test passes; wave 4 assembles **3,352 households
→ a 133-column one-hot matrix**, and the stacked ensemble scores ~0.98–0.99
accuracy / ~0.999 AUC against the K-Means labels. See the honest reading of that
number in the next section.

---

## What the ~99% accuracy actually means (read this before quoting it)

The very high accuracy is **expected and is not evidence that the model predicts
poverty well.** The target the classifier is trained on is the K-Means cluster
label, and that label is computed from the same features (scale → PCA →
nearest centroid). So the classifier is really being asked to *re-draw a
boundary that is already a deterministic function of its inputs* — a circular
setup. I checked this directly (`leakage_probe.py`):

- **Train vs test gap is tiny** (~0.6–1.5 pp), so this is not ordinary
  overfitting — the map genuinely generalizes.
- **Proper 5-fold CV** (K-Means label regenerated inside each fold's training
  data) still gives **0.989 ± 0.003 accuracy, 0.9995 AUC.** The number survives
  out-of-sample precisely *because* the target is self-generated, not because
  the model learned something about real-world poverty.
- **Control:** swap in a random label that is unrelated to the features and
  accuracy collapses to **~0.50 (chance).** So essentially 100% of the ~99%
  comes from the label being a function of the features.

Bottom line: this pipeline faithfully reproduces the reported >0.9 accuracy, but
that accuracy measures *"can the ensemble reconstruct the K-Means partition"*,
which is close to tautological. To claim real poverty-prediction skill I'd need
an **independent** ground-truth label (e.g. an Alkire-Foster / MPI or
consumption-based poverty line) and score against that — the hook for it is
`config.experiment.true_label_col`, which pulls the target out of the features
and drops it from the design matrix so it can't leak.

---

## Module layout

```
baseline_replication/
├── config.py              # ALL tunable knobs (one dataclass tree). Edit this first.
├── run_baseline.py        # CLI: assemble a wave → run pipeline → export results
├── smoke_test.py          # synthetic end-to-end check (no real data needed)
├── leakage_probe.py       # one-off circularity diagnostic (safe to delete)
├── data/
│   ├── feature_spec.py    # the ~28 features ↔ raw NPS columns (source of truth)
│   ├── loader.py          # join Sections A/B/C/I, head-of-household, hh size
│   └── encode.py          # one-hot encode categoricals → numeric design matrix
└── pipeline/
    ├── imputation.py      # KNN imputation                    ("KNN")
    ├── reduction.py       # PCA (or "none" to skip)           ("PCA")
    ├── clustering.py      # K-Means + optional silhouette K-search  ("K-Means")
    ├── nn.py              # PyTorch MLP + skorch wrapper (NN base learner)
    ├── stacking.py        # sklearn StackingClassifier        ("Stacked Ensemble")
    ├── metrics.py         # accuracy, macro P/R/F1/Jaccard, kappa, MCC, AUC
    └── experiment.py      # 20-seed loop, aggregation, Excel/plot export
```

---

## Design decisions worth knowing

1. **No leakage in the transforms.** The train/test split happens *first*, then
   the KNN imputer, the StandardScaler, and PCA are all fit on the training rows
   only and merely `transform` the test rows. K-Means is likewise fit on the
   training PCA space, and test rows are assigned to the nearest existing
   centroid. (Verified by reading `experiment.py` and re-running the steps
   independently in `leakage_probe.py`.)

2. **K = 2.** Two clusters, read as poor / non-poor. `config.cluster.auto_select_k`
   turns on silhouette-based selection over a range if I want to test other K.

3. **Classifier trains on the raw (imputed) features by default.**
   `config.stacking.classifier_input="raw"`. Switching it to `"reduced"` hands
   the classifier the same 2 PCA coordinates K-Means used, which makes the label
   almost trivially recoverable (test accuracy ≈ 1.0 in my probe) — useful to
   demonstrate the circularity, not for a real evaluation.

4. **Supervised target.** With no independent label configured, the target is
   the K-Means `cluster_label` (see the accuracy caveat above). Set
   `config.experiment.true_label_col` to score against a real label instead.

5. **Stacking composition.** Base learners span the same families as the
   reference stack — `glm`→LogisticRegression, `gbm`→GradientBoosting,
   `drf`→RandomForest, `xgb`→XGBClassifier (falls back to GradientBoosting if
   `xgboost` is absent), `nn`→the skorch MLP. Meta-learner is LogisticRegression.

---

## What I expect to tweak next (the "specific data parameters" phase)

- **`data/feature_spec.py`** — the exact feature set, and the value-code
  harmonizations (`hh_c07` education, `hh_i10` floor, `hh_i19` water) flagged
  `verify=True` for cross-wave use. For a single wave (W4/W5) they're fine as-is.
- **`config.cluster`** — K, and the label-ordering convention (which cluster is "poor").
- **`config.reduction`** — number of PCA components.
- **`config.stacking` / `config.nn`** — base-learner set, NN architecture/epochs.

SHAP is deliberately **not** here yet — it slots in after this baseline is
locked, ideally on the `classifier_input="raw"` variant so attributions map to
real features rather than principal components.