"""Command-line entry point: fit a wave, run SHAP, save the Week-3 deliverables.

Run it from the project root (the folder that holds `shap_analysis/` and
`baseline_replication/`):

    python -m shap_analysis.run_shap --wave 4
    python -m shap_analysis.run_shap --wave 5
    python -m shap_analysis.run_shap --wave 4 --full          # explain every household (slow)
    python -m shap_analysis.run_shap --wave 4 --no-zanzibar   # 6 mainland zones only

Outputs (to shap_analysis/outputs/):
    shap_values_wave{N}.csv        per-household SHAP over the one-hot columns
    shap_by_feature_wave{N}.csv    per-household |SHAP| aggregated to 18 features (+ zone)
    zone_importance_wave{N}.xlsx   zone x feature mean|SHAP|  (the SHAP matrix deliverable)
    zone_importance_wave{N}_heatmap.png           the zone-importance heatmap deliverable
    zone_importance_wave{N}_no_geo_heatmap.png    heatmap WITHOUT region/rural_urban (tautology check)
    feature_importance_wave{N}.png                overall feature-importance bar chart
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from .config import default_config
from .explain import run_shap
from .model import fit_full_model

# Geographic features to exclude in the "no-geo" variant heatmap.
_GEO_FEATURES = ["region", "rural_urban"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Zone-stratified SHAP for the poverty model.")
    p.add_argument("--wave", type=int, default=4, choices=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=None, help="Single seed for the fitted model.")
    p.add_argument("--full", action="store_true",
                   help="Explain EVERY household (slow; default samples per zone).")
    p.add_argument("--per-zone", type=int, default=None,
                   help="Households explained per zone in sampled mode (default 40).")
    p.add_argument("--background", type=int, default=None,
                   help="k-means background summary size for SHAP (default 50).")
    p.add_argument("--no-zanzibar", action="store_true",
                   help="Restrict to the 6 mainland zones (drop Zanzibar households).")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def build_config_from_args(args):
    cfg = default_config()
    cfg.wave = args.wave
    if args.seed is not None:
        cfg.seed = args.seed
    if args.full:
        cfg.full = True
    if args.per_zone is not None:
        cfg.per_zone_sample = args.per_zone
    if args.background is not None:
        cfg.background_size = args.background
    if args.no_zanzibar:
        cfg.include_zanzibar = False
    if args.no_plots:
        cfg.make_plots = False
    return cfg


def _report_clusters(model) -> None:
    counts = pd.Series(model.labels).value_counts().sort_index()
    total = int(counts.sum())
    print("  K-Means cluster sizes (label 1 = more-deprived / 'poor'):")
    for label, n in counts.items():
        print(f"    cluster {label}: {n:5d}  ({100 * n / total:5.1f}%)")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    mode = "FULL (all households)" if cfg.full else f"sampled (<= {cfg.per_zone_sample}/zone)"
    print(f"Wave {cfg.wave} | seed {cfg.seed} | SHAP mode: {mode}")

    print("Fitting pipeline (KNN impute -> PCA -> K-Means -> Stacked Ensemble) ...")
    t0 = time.time()
    model = fit_full_model(cfg)
    print(f"  households: {len(model.X_imp)}  one-hot features: {len(model.feature_columns)}"
          f"  (fit {time.time() - t0:.1f}s)")
    _report_clusters(model)

    # ---- Main SHAP run (all features) ------------------------------------ #
    print("Running SHAP KernelExplainer on P(poor) of the raw-feature model ...")
    t1 = time.time()
    result = run_shap(model, cfg)
    print(f"  explained {len(result.foreground_index)} households"
          f"  ({time.time() - t1:.1f}s)")
    print(f"  SHAP base value (mean P(poor)): {result.expected_value:.3f}")

    # ---- Per-zone sample sizes ------------------------------------------- #
    print("\n  Per-zone household counts:")
    counts_df = pd.DataFrame({
        "zone": result.zone_total_counts.index,
        "n_total": result.zone_total_counts.values,
        "n_explained": result.zone_explained_counts.values,
    })
    for _, row in counts_df.iterrows():
        print(f"    {row['zone']:12s}  total={int(row['n_total']):5d}  "
              f"explained={int(row['n_explained']):4d}")

    # ---- No-geo variant (exclude region + rural_urban from aggregation) --- #
    # Reuse the already-computed per-household |SHAP| — only the zone-level
    # aggregation changes, so there is no need to re-run the expensive explainer.
    print("\nComputing no-geo variant (excluding region, rural_urban from aggregation) ...")
    non_geo_cols = [c for c in result.zone_importance.columns if c not in _GEO_FEATURES]
    zone_importance_no_geo = result.shap_by_feature.groupby("zone")[non_geo_cols].mean().reindex(
        result.zone_importance.index
    )

    # ---- Save artifacts -------------------------------------------------- #
    w = cfg.wave
    od = cfg.output_dir
    result.shap_onehot.to_csv(od / f"shap_values_wave{w}.csv")
    result.shap_by_feature.to_csv(od / f"shap_by_feature_wave{w}.csv", index=False)

    # Per-zone count of households actually aggregated in these SHAP matrices.
    # Added as an `n_households` column on the zone_x_feature sheets so a reader
    # can see how many households back each zone's mean|SHAP| row without opening
    # the separate zone_sample_sizes sheet. Inserted only into the Excel copies —
    # the DataFrames handed to the plotter stay clean so no count column leaks
    # into the heatmaps.
    n_hh = counts_df.set_index("zone")["n_explained"]
    zi_main_xlsx = result.zone_importance.copy()
    zi_main_xlsx.insert(0, "n_households",
                        n_hh.reindex(zi_main_xlsx.index).to_numpy())
    zi_ng_xlsx = zone_importance_no_geo.copy()
    zi_ng_xlsx.insert(0, "n_households",
                      n_hh.reindex(zi_ng_xlsx.index).to_numpy())

    # Main Excel with all-features zone importance + zone sample sizes.
    xlsx = od / f"zone_importance_wave{w}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        zi_main_xlsx.to_excel(writer, sheet_name="zone_x_feature")
        # A tidy long form is easier to filter in a dashboard later.
        (result.zone_importance.reset_index()
         .melt(id_vars="zone", var_name="feature", value_name="mean_abs_shap")
         .to_excel(writer, sheet_name="long", index=False))
        # Per-zone household counts.
        counts_df.to_excel(writer, sheet_name="zone_sample_sizes", index=False)
        pd.DataFrame({
            "setting": ["wave", "seed", "mode", "per_zone_sample", "background_size",
                        "include_zanzibar", "explain_class", "n_explained",
                        "expected_value"],
            "value": [w, cfg.seed, "full" if cfg.full else "sampled",
                      cfg.per_zone_sample, cfg.background_size, cfg.include_zanzibar,
                      cfg.explain_class, len(result.foreground_index),
                      result.expected_value],
        }).to_excel(writer, sheet_name="run_config", index=False)
    print(f"  saved SHAP matrix -> {xlsx}")

    # No-geo Excel.
    xlsx_ng = od / f"zone_importance_wave{w}_no_geo.xlsx"
    with pd.ExcelWriter(xlsx_ng, engine="openpyxl") as writer:
        zi_ng_xlsx.to_excel(writer, sheet_name="zone_x_feature")
        (zone_importance_no_geo.reset_index()
         .melt(id_vars="zone", var_name="feature", value_name="mean_abs_shap")
         .to_excel(writer, sheet_name="long", index=False))
        counts_df.to_excel(writer, sheet_name="zone_sample_sizes", index=False)
    print(f"  saved no-geo SHAP matrix -> {xlsx_ng}")

    print("\nZone x feature mean|SHAP| (rounded, all features):")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(result.zone_importance.round(3).to_string())

    print("\nZone x feature mean|SHAP| (rounded, WITHOUT geographic features):")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(zone_importance_no_geo.round(3).to_string())

    if cfg.make_plots:
        from .plots import global_importance_bar, zone_importance_heatmap
        hm = zone_importance_heatmap(result.zone_importance, w,
                                     od / f"zone_importance_wave{w}_heatmap.png")
        hm_ng = zone_importance_heatmap(zone_importance_no_geo, w,
                                        od / f"zone_importance_wave{w}_no_geo_heatmap.png")
        bar = global_importance_bar(result.zone_importance, w,
                                    od / f"feature_importance_wave{w}.png")
        print(f"  saved heatmap          -> {hm}")
        print(f"  saved no-geo heatmap   -> {hm_ng}")
        print(f"  saved bar chart        -> {bar}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

