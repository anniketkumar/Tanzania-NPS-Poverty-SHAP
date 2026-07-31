"""Cross-wave SHAP comparison: how feature importance drifts across NPS waves.

Reads the per-wave zone_importance deliverables produced by `run_shap.py` and
assembles a combined feature x wave table of overall (zone-averaged) mean|SHAP|,
for both the full-feature and no-geo variants, plus a slope plot of the top
features. Automatically detects which waves (1–5) have outputs available.

"Overall" importance is aggregated exactly as the Week-3 bar chart
(`plots.global_importance_bar`): the unweighted mean across zones of each zone's
mean|SHAP|. So these numbers line up with `feature_importance_wave{N}.png`.

    python -m shap_analysis.compare_waves

Outputs (to shap_analysis/outputs/):
    wave_comparison_full.xlsx / .csv       feature x wave, all features
    wave_comparison_no_geo.xlsx / .csv     feature x wave, region+rural_urban dropped
    wave_comparison_full.png               slope plot, top features
    wave_comparison_no_geo.png             slope plot, top features (no geo)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import default_config

ALL_WAVES = (1, 2, 3, 4, 5)
WAVE_YEARS = {1: "2008/09", 2: "2010/11", 3: "2012/13", 4: "2014/15", 5: "2020/21"}
# Features to call out in the plot — the drift the Week-4 question is about.
HIGHLIGHT = ["electricity_source", "rural_urban"]
OD = default_config().output_dir


def _available_waves(no_geo: bool) -> tuple[int, ...]:
    """Return waves whose SHAP outputs exist on disk."""
    suffix = "_no_geo" if no_geo else ""
    return tuple(w for w in ALL_WAVES
                 if (OD / f"zone_importance_wave{w}{suffix}.xlsx").is_file())


def _overall_importance(wave: int, no_geo: bool) -> pd.Series:
    """Zone-averaged mean|SHAP| per feature for one wave (matches the bar chart)."""
    suffix = "_no_geo" if no_geo else ""
    path = OD / f"zone_importance_wave{wave}{suffix}.xlsx"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m shap_analysis.run_shap --wave {wave}` first."
        )
    zi = pd.read_excel(path, sheet_name="zone_x_feature", index_col=0)
    zi = zi.drop(columns=[c for c in ("n_households",) if c in zi.columns])
    # Unweighted mean across zones = the "overall feature importance".
    return zi.mean(axis=0).rename(f"W{wave} ({WAVE_YEARS[wave]})")


def _zone_counts(wave: int) -> pd.Series:
    path = OD / f"zone_importance_wave{wave}.xlsx"
    if not path.is_file():
        return pd.Series(dtype=float)
    zi = pd.read_excel(path, sheet_name="zone_x_feature", index_col=0)
    return zi["n_households"].rename(f"W{wave}") if "n_households" in zi.columns else pd.Series(dtype=float)


def build_table(no_geo: bool) -> pd.DataFrame:
    waves = _available_waves(no_geo)
    if not waves:
        raise FileNotFoundError("No SHAP outputs found. Run run_shap.py first.")
    cols = [_overall_importance(w, no_geo) for w in waves]
    table = pd.concat(cols, axis=1)
    # Rank by the largest importance any wave assigns the feature.
    # NaN cells are expected when a feature is absent in some waves
    # (e.g. child_stool_disposal missing from W1/W2).
    table = table.reindex(table.max(axis=1).sort_values(ascending=False).index)
    return table, waves


def slope_plot(table: pd.DataFrame, waves: tuple[int, ...],
               out_path: Path, title: str, top_n: int = 10) -> Path:
    top = table.head(top_n)
    xs = list(range(len(waves)))
    xlabels = [f"W{w}\n{WAVE_YEARS[w]}" for w in waves]

    fig, ax = plt.subplots(figsize=(9 + len(waves) * 0.5, 6))
    for feat, row in top.iterrows():
        vals = row.values
        if feat in HIGHLIGHT:
            ax.plot(xs, vals, marker="o", linewidth=2.6, zorder=3,
                    label=feat)
        else:
            ax.plot(xs, vals, marker="o", linewidth=1.2, alpha=0.55,
                    color="#9aa0a6", zorder=1)
            # Find the last non-NaN value for annotation.
            last_valid = row.last_valid_index()
            if last_valid is not None:
                ix = list(row.index).index(last_valid)
                ax.annotate(feat, (ix, row[last_valid]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color="#5f6368")
    # Label highlighted lines at the left end too.
    for feat in HIGHLIGHT:
        if feat in top.index:
            first_valid = top.loc[feat].first_valid_index()
            if first_valid is not None:
                ix = list(top.columns).index(first_valid)
                ax.annotate(feat, (ix, top.loc[feat, first_valid]), xytext=(-6, 0),
                            textcoords="offset points", va="center", ha="right",
                            fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("overall mean |SHAP| (zone-averaged)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="highlighted", loc="upper left", fontsize=9)
    ax.margins(x=0.15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _save_table(table: pd.DataFrame, waves: tuple[int, ...], stem: str) -> None:
    table.to_csv(OD / f"{stem}.csv")
    counts = pd.concat([_zone_counts(w) for w in waves], axis=1)
    with pd.ExcelWriter(OD / f"{stem}.xlsx", engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="feature_x_wave")
        if not counts.empty:
            counts.to_excel(writer, sheet_name="n_households_per_zone")


def main() -> int:
    for no_geo, stem, title in [
        (False, "wave_comparison_full",
         "SHAP feature-importance drift across NPS waves (all features)"),
        (True, "wave_comparison_no_geo",
         "SHAP feature-importance drift across NPS waves (no geo)"),
    ]:
        table, waves = build_table(no_geo)
        n_waves = len(waves)
        wave_str = ", ".join(f"W{w}" for w in waves)
        print(f"\n=== {stem} ({n_waves} waves: {wave_str}) ===")
        _save_table(table, waves, stem)
        png = slope_plot(table, waves, OD / f"{stem}.png", title)
        with pd.option_context("display.width", 200):
            print(table.round(4).to_string())
        print(f"  saved -> {OD / (stem + '.csv')}, {OD / (stem + '.xlsx')}, {png}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
