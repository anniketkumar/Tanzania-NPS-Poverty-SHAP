"""All the knobs for the Week-3 SHAP analysis in one place.

Same philosophy as `baseline_replication/config.py`: every setting lives here so
that adapting a run means editing this one file rather than digging through the
modules. The pipeline settings themselves (KNN k, PCA components, K-Means K, the
stacking composition) are pulled from the *baseline* config so this layer stays
faithful to the locked Week 1-2 replication — I only add the SHAP-specific knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from baseline_replication.config import Config as BaselineConfig
from baseline_replication.config import default_config as baseline_default_config

# Outputs land next to this package, in shap_analysis/outputs/.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class SHAPConfig:
    """Everything the SHAP layer needs on top of the baseline pipeline."""

    # --- what to explain -------------------------------------------------- #
    wave: int = 4                       # NPS wave to explain (3, 4 or 5)
    seed: int = 42                      # single seed: SHAP needs ONE fitted model

    # --- SHAP compute budget --------------------------------------------- #
    # KernelExplainer is model-agnostic (needed here — the stack has an MLP + a
    # logistic meta-learner, so TreeExplainer doesn't apply) but slow. So by
    # default I summarise the background with k-means and only explain a
    # stratified per-zone SAMPLE of households. `full=True` explains everyone.
    background_size: int = 50           # k-means summary rows for the SHAP background
    per_zone_sample: int = 30           # households explained per zone (sampled mode)
    full: bool = False                  # explain every household (slow, publication run)
    # KernelExplainer perturbations per row. "auto" (=2*n_features+2048) is ~6.8s
    # /row here; a moderate fixed budget is ~2.5x faster and, because zone
    # importance AVERAGES over many households, gives an indistinguishable matrix.
    nsamples: str | int = 500
    explain_class: int = 1              # which class to attribute (1 = "poor" cluster)

    # --- zones ------------------------------------------------------------ #
    # The paper's taxonomy has 7 zones INCLUDING Zanzibar. Keep it on by default;
    # set False to restrict to the 6 mainland zones.
    include_zanzibar: bool = True

    # --- output ----------------------------------------------------------- #
    output_dir: Path = OUTPUT_DIR
    make_plots: bool = True

    # --- baseline pipeline settings (inherited, not re-tuned here) -------- #
    baseline: BaselineConfig = field(default_factory=baseline_default_config)

    def synced_baseline(self) -> BaselineConfig:
        """Return the baseline config with wave/seed aligned to this run.

        The classifier must train on the RAW (non-PCA) imputed features so SHAP
        values map to real survey questions — that is already the baseline
        default (`classifier_input="raw"`), and I assert it here so a change to
        the baseline can't silently push SHAP onto principal components.
        """
        cfg = self.baseline
        cfg.data.wave = self.wave
        assert cfg.stacking.classifier_input == "raw", (
            "SHAP must explain the raw-feature model; set baseline "
            "config.stacking.classifier_input = 'raw'."
        )
        return cfg


def default_config() -> SHAPConfig:
    return SHAPConfig()
