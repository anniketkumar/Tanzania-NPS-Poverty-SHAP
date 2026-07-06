"""All the knobs for the pipeline in one place.

I keep every setting here so that when I need to adapt this to my own data or
try a different value, I edit this one file instead of digging through the
modules. Each dataclass below matches one stage of the pipeline.

A few places where I deliberately differ from the tools the original paper used,
because I'm rebuilding this in a plain scikit-learn / PyTorch stack:
  - PCA for dimensionality reduction (my source is PCA-only).
  - scikit-learn's StackingClassifier instead of H2O AutoML's stacked ensemble.
  - a small PyTorch MLP (wrapped by skorch) instead of H2O's deep-learning model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

# Repository root = the folder that contains both `aniket_code/` and `converted data/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTED_DATA_DIR = PROJECT_ROOT / "converted data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class DataConfig:
    """Which survey wave to build and where the raw CSVs are."""

    wave: int = 4                       # the wave I'm replicating (4 or 5)
    converted_data_dir: Path = CONVERTED_DATA_DIR
    # Collapse each household down to a single row: its head. In these surveys
    # the head is the member whose index is 1 (indidy{wave} == 1).
    head_of_household_only: bool = True
    # If True, throw away any row that still has a missing value before I impute.
    # I leave this off because the whole point of the KNN step is to fill those
    # gaps instead of losing rows. Flip it on if I want to compare against a
    # simple drop-missing approach.
    dropna_before_impute: bool = False


@dataclass
class ImputeConfig:
    """KNN imputation — the 'KNN' step of KNN -> PCA -> K-Means -> Stacked Ensemble."""

    enabled: bool = True
    n_neighbors: int = 5                # how many neighbours to average when filling a gap
    weights: str = "uniform"            # "uniform" = equal weight; "distance" = closer counts more


@dataclass
class ReductionConfig:
    """PCA settings for the reduction step that feeds K-Means.

    My source only uses PCA, so that's the default and the intended path. I
    reduce to 2 components so the clustering happens in a small, denoised space.
    "none" is just a debugging option that skips PCA entirely.
    """

    method: str = "pca"                 # "pca" | "none"
    n_components: int = 2               # 2 components going into K-Means
    standardize: bool = True            # scale features before PCA (PCA is scale-sensitive)


@dataclass
class ClusterConfig:
    """K-Means — this is what actually *creates* the poor/non-poor label."""

    n_clusters: int = 2                 # two clusters: poor vs non-poor
    n_init: int = 10                    # restart K-Means 10 times and keep the best fit
    # If True, ignore n_clusters and pick K automatically by silhouette score.
    auto_select_k: bool = False
    k_search_range: tuple[int, ...] = (2, 3, 4, 5)
    # K-Means numbers its clusters arbitrarily (which group is "0" changes with
    # the seed). This re-labels them by how far the centroid sits from the
    # origin in PCA space so that "0" and "1" mean the same thing every run,
    # which is what makes per-class metrics comparable across seeds.
    order_labels_by_deprivation: bool = True


@dataclass
class StackingConfig:
    """scikit-learn StackingClassifier standing in for H2O's stacked ensemble.

    I picked base learners that cover the same model families the paper stacked
    (a linear model, two tree ensembles, boosting, and a neural net):
        glm  -> LogisticRegression
        gbm  -> GradientBoostingClassifier
        drf  -> RandomForestClassifier
        xgb  -> XGBClassifier  (falls back to GradientBoosting if xgboost is missing)
        nn   -> skorch NeuralNetClassifier wrapping a PyTorch MLP
    The meta-learner that combines them is a LogisticRegression.
    """

    base_learners: tuple[str, ...] = ("glm", "gbm", "drf", "xgb", "nn")
    meta_learner: str = "glm"           # "glm" | "gbm"
    cv: int = 5                         # folds the stacker uses internally to build meta-features
    stack_method: str = "predict_proba"
    passthrough: bool = False           # also give the meta-learner the raw features?
    n_jobs: int = 1                     # keep at 1 — torch + parallel process pools can hang

    # Which feature space the classifier trains on:
    #   "raw"     -> the imputed engineered features (default)
    #   "reduced" -> the same 2 PCA coordinates K-Means used
    # NOTE: "reduced" makes the label almost trivially recoverable, since the
    # classifier then sees the exact coordinates the label was drawn from.
    classifier_input: str = "raw"


@dataclass
class NNConfig:
    """Hyper-parameters for the PyTorch MLP (the skorch-wrapped base learner)."""

    hidden_dims: tuple[int, ...] = (32, 16)
    dropout: float = 0.1
    max_epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    device: str = "cpu"                 # switch to "cuda" if a GPU is around


@dataclass
class ExperimentConfig:
    """The loop that repeats the whole pipeline over many random seeds."""

    # 20 seeds (42..61); I average metrics over all of them to smooth out the
    # luck of any single train/test split.
    seeds: tuple[int, ...] = tuple(range(42, 62))
    test_size: float = 0.30
    # If I ever have a real, independent poverty label (e.g. an Alkire-Foster /
    # MPI label), I put its column name here and the classifier is scored
    # against THAT instead of the K-Means label. When None, the target is the
    # K-Means cluster label itself (see the note in experiment.py about why that
    # makes the accuracy number circular).
    true_label_col: str | None = None
    output_dir: Path = OUTPUT_DIR
    excel_out: str = "experimental_summary.xlsx"
    make_plots: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    impute: ImputeConfig = field(default_factory=ImputeConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def default_config() -> Config:
    return Config()
