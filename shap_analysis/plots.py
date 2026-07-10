"""Turn the zone x feature SHAP matrix into the policy-facing heatmap.

The heatmap is the visual half of the Week-3 deliverable: rows are Tanzania's
zones, columns are survey features, and each cell is the mean |SHAP| — i.e. how
much that feature drives the poor/non-poor classification in that zone. Reading
down a column shows which zones a driver matters most in; reading across a row
answers the proposal's question: *"in the Western zone, should limited budget go
to electrification, schools, or water?"*
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")               # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def zone_importance_heatmap(
    zone_importance: pd.DataFrame,
    wave: int,
    out_path: Path,
    sort_columns: bool = True,
) -> Path:
    """Save an annotated zone x feature mean|SHAP| heatmap and return its path."""
    mat = zone_importance.copy()
    if sort_columns and not mat.empty:
        # Put the globally most important features on the left.
        col_order = mat.mean(axis=0).sort_values(ascending=False).index
        mat = mat[col_order]

    n_zones, n_feats = mat.shape
    fig, ax = plt.subplots(figsize=(max(10, n_feats * 0.8), max(4, n_zones * 0.7)))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="rocket_r",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "mean |SHAP| (contribution to P(poor))"},
    )
    ax.set_title(
        f"Zone-stratified feature importance (SHAP) — NPS Wave {wave}",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Survey feature")
    ax.set_ylabel("Zone")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def global_importance_bar(
    zone_importance: pd.DataFrame,
    wave: int,
    out_path: Path,
) -> Path:
    """Save a bar chart of overall (zone-averaged) feature importance."""
    order = zone_importance.mean(axis=0).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(order) * 0.35)))
    ax.barh(order.index, order.values, color="#4c72b0")
    ax.set_title(f"Overall feature importance (mean |SHAP| across zones) — Wave {wave}")
    ax.set_xlabel("mean |SHAP|")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
