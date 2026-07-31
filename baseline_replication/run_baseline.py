"""Command-line entry point: build a wave, run the pipeline, save the results.

Run it from the project root (the folder that holds `baseline_replication/`):

    python -m baseline_replication.run_baseline --wave 4
    python -m baseline_replication.run_baseline --wave 5 --k 2
    python -m baseline_replication.run_baseline --wave 4 --seeds 5     # quick 5-seed run
    python -m baseline_replication.run_baseline --wave 1              # Wave 1 (2008/09)

All five NPS waves (1–5) are supported. Waves 1 and 2 use different column
prefixes and file layouts, handled by `data/wave_config.py`. Value codes for
all waves are harmonized to the W4/W5 reference scheme in `data/harmonize.py`.
Each wave writes to its own output files (experimental_summary_wave{N}.xlsx).

Note: Waves 1 and 2 have 17 features (child_stool_disposal was not asked).
"""

from __future__ import annotations

import argparse
import sys

from .config import default_config
from .data import build_design_matrix, load_wave_features
from .pipeline import run_experiment


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baseline poverty-clustering pipeline.")
    p.add_argument("--wave", type=int, default=4, choices=[1, 2, 3, 4, 5])
    p.add_argument("--reducer", default=None,
                   choices=["pca", "none"],
                   help="Override the reduction method (pca, or none to skip it).")
    p.add_argument("--k", type=int, default=None, help="Number of K-Means clusters.")
    p.add_argument("--seeds", type=int, default=None,
                   help="Use only the first N seeds (handy for quick runs).")
    p.add_argument("--classifier-input", default=None, choices=["raw", "reduced"])
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def build_config_from_args(args):
    cfg = default_config()
    cfg.data.wave = args.wave
    # Per-wave output filenames so running one wave doesn't overwrite another's
    # results (the old fixed names shared a single file across waves).
    cfg.experiment.excel_out = f"experimental_summary_wave{args.wave}.xlsx"
    if args.reducer is not None:
        cfg.reduction.method = args.reducer
    if args.k is not None:
        cfg.cluster.n_clusters = args.k
    if args.seeds is not None:
        cfg.experiment.seeds = tuple(cfg.experiment.seeds[: args.seeds])
    if args.classifier_input is not None:
        cfg.stacking.classifier_input = args.classifier_input
    if args.no_plots:
        cfg.experiment.make_plots = False
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config_from_args(args)

    print(f"Assembling wave {cfg.data.wave} features ...")
    raw = load_wave_features(
        cfg.data.converted_data_dir, cfg.data.wave, cfg.data.head_of_household_only
    )
    print(f"  households: {len(raw)}  raw feature columns: {raw.shape[1] - 1}")

    design = build_design_matrix(raw)
    if cfg.data.dropna_before_impute:
        before = len(design)
        design = design.dropna()
        print(f"  dropped {before - len(design)} rows with missing values (dropna mode)")
    print(f"  design matrix: {design.shape[0]} x {design.shape[1]} "
          f"(missing cells: {int(design.isna().sum().sum())})")

    # By default there's no independent poverty label, so the target ends up
    # being the K-Means cluster. If I've configured a real label column, pull it
    # out here and drop it from the features so it can't leak into training.
    y_ext = None
    if cfg.experiment.true_label_col and cfg.experiment.true_label_col in raw.columns:
        y_ext = raw[cfg.experiment.true_label_col].to_numpy()
        design = design.drop(columns=[cfg.experiment.true_label_col], errors="ignore")

    run_experiment(design, y_ext, cfg)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
