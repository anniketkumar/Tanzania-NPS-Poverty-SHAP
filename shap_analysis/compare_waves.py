"""3-wave SHAP comparison: how feature importance drifts W3 -> W4 -> W5.

Reads the per-wave zone_importance deliverables produced by `run_shap.py` and
assembles a combined feature x wave table of overall (zone-averaged) mean|SHAP|,
for both the full-feature and no-geo variants, plus a slope plot of the top
features across the three waves.

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

WAVES = (3, 4, 5)
WAVE_YEARS = {3: "2012/13", 4: "2014/15", 5: "2020/21"}
# Features to call out in the plot — the drift the Week-4 question is about.
HIGHLIGHT = ["electricity_source", "rural_urban"]
OD = default_config().output_dir


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
    zi = pd.read_excel(OD / f"zone_importance_wave{wave}.xlsx",
                       sheet_name="zone_x_feature", index_col=0)
    return zi["n_households"].rename(f"W{wave}") if "n_households" in zi.columns else pd.Series(dtype=float)


def build_table(no_geo: bool) -> pd.DataFrame:
    cols = [_overall_importance(w, no_geo) for w in WAVES]
    table = pd.concat(cols, axis=1)
    # Rank by the largest importance any wave assigns the feature.
    table = table.reindex(table.max(axis=1).sort_values(ascending=False).index)
    return table


def slope_plot(table: pd.DataFrame, out_path: Path, title: str, top_n: int = 10) -> Path:
    top = table.head(top_n)
    xs = list(range(len(WAVES)))
    xlabels = [f"W{w}\n{WAVE_YEARS[w]}" for w in WAVES]

    fig, ax = plt.subplots(figsize=(9, 6))
    for feat, row in top.iterrows():
        if feat in HIGHLIGHT:
            ax.plot(xs, row.values, marker="o", linewidth=2.6, zorder=3,
                    label=feat)
        else:
            ax.plot(xs, row.values, marker="o", linewidth=1.2, alpha=0.55,
                    color="#9aa0a6", zorder=1)
            ax.annotate(feat, (xs[-1], row.values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color="#5f6368")
    # Label highlighted lines at the left end too.
    for feat in HIGHLIGHT:
        if feat in top.index:
            ax.annotate(feat, (xs[0], top.loc[feat].values[0]), xytext=(-6, 0),
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


def _save_table(table: pd.DataFrame, stem: str) -> None:
    table.to_csv(OD / f"{stem}.csv")
    counts = pd.concat([_zone_counts(w) for w in WAVES], axis=1)
    with pd.ExcelWriter(OD / f"{stem}.xlsx", engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="feature_x_wave")
        counts.to_excel(writer, sheet_name="n_households_per_zone")


def main() -> int:
    for no_geo, stem, title in [
        (False, "wave_comparison_full",
         "SHAP feature-importance drift across NPS waves (all features)"),
        (True, "wave_comparison_no_geo",
         "SHAP feature-importance drift across NPS waves (no geo)"),
    ]:
        table = build_table(no_geo)
        _save_table(table, stem)
        png = slope_plot(table, OD / f"{stem}.png", title)
        print(f"\n=== {stem} ===")
        with pd.option_context("display.width", 160):
            print(table.round(4).to_string())
        print(f"  saved -> {OD / (stem + '.csv')}, {OD / (stem + '.xlsx')}, {png}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
