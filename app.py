"""
Tanzania NPS Poverty SHAP Dashboard
====================================
Streamlit dashboard for the zone-stratified SHAP analysis of Tanzania's
National Panel Survey (2008/09 – 2020/21).  Reads ONLY from pre-computed
output files produced by the locked baseline_replication and shap_analysis
pipelines — no model fitting or SHAP computation happens here.

Launch:
    pip install -r requirements_dashboard.txt
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BASELINE_OUT = ROOT / "baseline_replication" / "outputs"
SHAP_OUT = ROOT / "shap_analysis" / "outputs"

ALL_WAVES = [1, 2, 3, 4, 5]
WAVE_YEARS = {1: "2008/09", 2: "2010/11", 3: "2012/13", 4: "2014/15", 5: "2020/21"}
ZONE_ORDER = ["Western", "Lake", "Central", "Southern", "Northern", "Coastal", "Zanzibar"]
HIGHLIGHT_FEATURES = ["electricity_source", "rural_urban"]

# ---------------------------------------------------------------------------
# Data loading helpers (cached so files are read once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_experimental_summary(wave: int) -> dict:
    """Load aggregate metrics and run config from the baseline experimental summary."""
    path = BASELINE_OUT / f"experimental_summary_wave{wave}.xlsx"
    if not path.is_file():
        return {}
    agg = pd.read_excel(path, sheet_name="aggregate")
    info: dict = {}
    # The aggregate sheet has metric / mean / std columns.
    for _, row in agg.iterrows():
        metric = str(row.iloc[0]).strip().lower()
        if "accuracy" in metric and "train" not in metric:
            info["accuracy_mean"] = row.iloc[1]
            info["accuracy_std"] = row.iloc[2] if len(row) > 2 else 0
        elif "auc" in metric:
            info["auc_mean"] = row.iloc[1]
    # Try to get base rate from the SHAP run_config sheet.
    shap_path = SHAP_OUT / f"zone_importance_wave{wave}.xlsx"
    if shap_path.is_file():
        try:
            rc = pd.read_excel(shap_path, sheet_name="run_config")
            for _, row in rc.iterrows():
                if str(row["setting"]).strip() == "expected_value":
                    info["base_rate"] = float(row["value"])
        except Exception:
            pass
    return info


@st.cache_data(show_spinner=False)
def load_zone_importance(wave: int, no_geo: bool) -> pd.DataFrame | None:
    """Zone × feature mean|SHAP| matrix."""
    suffix = "_no_geo" if no_geo else ""
    path = SHAP_OUT / f"zone_importance_wave{wave}{suffix}.xlsx"
    if not path.is_file():
        return None
    df = pd.read_excel(path, sheet_name="zone_x_feature", index_col=0)
    # Drop the n_households provenance column from the importance matrix.
    if "n_households" in df.columns:
        df = df.drop(columns=["n_households"])
    return df


@st.cache_data(show_spinner=False)
def load_zone_sample_sizes(wave: int) -> pd.DataFrame | None:
    """Per-zone total and explained household counts."""
    path = SHAP_OUT / f"zone_importance_wave{wave}.xlsx"
    if not path.is_file():
        return None
    try:
        return pd.read_excel(path, sheet_name="zone_sample_sizes")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_shap_by_feature(wave: int) -> pd.DataFrame | None:
    """Per-household |SHAP| aggregated to features, with zone column."""
    path = SHAP_OUT / f"shap_by_feature_wave{wave}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_wave_comparison(no_geo: bool) -> pd.DataFrame | None:
    """Feature × wave table of overall (zone-averaged) mean|SHAP|."""
    stem = "wave_comparison_no_geo" if no_geo else "wave_comparison_full"
    path = SHAP_OUT / f"{stem}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tanzania NPS – SHAP Poverty Explorer",
    page_icon="🇹🇿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — global controls + circularity caveat
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🇹🇿 NPS SHAP Explorer")
    st.caption("Explaining Poverty Clusters Across Space and Time")

    st.divider()

    st.warning(
        "**⚠️ Label-circularity caveat**\n\n"
        "The ~99% classification accuracy reflects reconstruction of a "
        "**K-Means-derived label**, not validated poverty prediction. "
        "The poor/non-poor target is generated by K-Means on the same "
        "features the classifier trains on. SHAP attributions explain "
        "*how the model draws the deprivation split*, which is the right "
        "object for policy analysis, but they are one step removed from "
        "independently-validated poverty.\n\n"
        "A random-label control collapses accuracy to ~50%, confirming "
        "the structural nature of this result. See Final Report §5 for "
        "full discussion.",
        icon="⚠️",
    )

    st.divider()
    st.caption(
        "Data: Tanzania National Panel Survey, Waves 1–5 (2008/09–2020/21).  \n"
        "Method: Sende et al. (2025) baseline + zone-stratified SHAP.  \n"
        "Dashboard reads pre-computed outputs only — no recomputation."
    )

# ---------------------------------------------------------------------------
# Main area — three tabs
# ---------------------------------------------------------------------------
tab_zone, tab_shap, tab_drift = st.tabs([
    "📍 Zone Selector",
    "🔬 SHAP Explorer",
    "📈 Temporal Drift",
])

# ============================= TAB 1: Zone Selector ========================
with tab_zone:
    st.header("Zone Selector")
    st.markdown("Select a wave and zone(s) to view poverty-cluster composition and key statistics.")

    col_wave, col_zones = st.columns([1, 3])
    with col_wave:
        sel_wave = st.selectbox(
            "NPS Wave",
            ALL_WAVES,
            index=3,  # default to Wave 4
            format_func=lambda w: f"Wave {w} ({WAVE_YEARS[w]})",
            key="zone_wave",
        )
    with col_zones:
        sel_zones = st.multiselect(
            "Zone(s)",
            ZONE_ORDER,
            default=ZONE_ORDER,
            key="zone_zones",
        )

    if not sel_zones:
        st.info("Select at least one zone above.")
    else:
        # --- Summary metrics ---
        info = load_experimental_summary(sel_wave)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            acc = info.get("accuracy_mean")
            acc_std = info.get("accuracy_std", 0)
            if acc is not None:
                # Handle both 0-1 and 0-100 formats
                if acc < 1:
                    st.metric("Accuracy (20 seeds)", f"{acc*100:.2f}% ± {acc_std*100:.2f}%")
                else:
                    st.metric("Accuracy (20 seeds)", f"{acc:.2f}% ± {acc_std:.2f}%")
            else:
                st.metric("Accuracy", "—")
        with m2:
            auc = info.get("auc_mean")
            if auc is not None:
                if auc < 1:
                    st.metric("AUC (macro)", f"{auc*100:.2f}%")
                else:
                    st.metric("AUC (macro)", f"{auc:.2f}%")
            else:
                st.metric("AUC", "—")
        with m3:
            br = info.get("base_rate")
            if br is not None:
                st.metric("Base rate P(poor)", f"{br:.3f}")
            else:
                st.metric("Base rate P(poor)", "—")
        with m4:
            st.metric("Wave", f"W{sel_wave} ({WAVE_YEARS[sel_wave]})")

        # --- Zone sample sizes ---
        st.subheader("Zone Composition")
        zss = load_zone_sample_sizes(sel_wave)
        if zss is not None:
            zss_filtered = zss[zss["zone"].isin(sel_zones)].copy()
            zss_filtered = zss_filtered.set_index("zone").reindex(
                [z for z in ZONE_ORDER if z in sel_zones]
            )
            if "n_total" in zss_filtered.columns and "n_explained" in zss_filtered.columns:
                zss_display = zss_filtered.rename(columns={
                    "n_total": "Total Households",
                    "n_explained": "SHAP-Explained Households",
                })
                st.dataframe(zss_display, use_container_width=True)

                # Bar chart of zone sizes
                fig_zones = px.bar(
                    zss_filtered.reset_index(),
                    x="zone",
                    y="n_total",
                    color="zone",
                    title=f"Household Count by Zone — Wave {sel_wave}",
                    labels={"n_total": "Total Households", "zone": "Zone"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_zones.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_zones, use_container_width=True)
            else:
                st.dataframe(zss_filtered, use_container_width=True)
        else:
            st.info("Zone sample-size data not available for this wave.")

        # --- Top features for selected zones ---
        st.subheader("Top Features by Zone (no-geo)")
        zi = load_zone_importance(sel_wave, no_geo=True)
        if zi is not None:
            zi_filtered = zi.loc[zi.index.isin(sel_zones)]
            if not zi_filtered.empty:
                # Show top 5 features per zone
                top_data = []
                for zone in zi_filtered.index:
                    row = zi_filtered.loc[zone].sort_values(ascending=False)
                    for rank, (feat, val) in enumerate(row.head(5).items(), 1):
                        top_data.append({
                            "Zone": zone,
                            "Rank": rank,
                            "Feature": feat,
                            "mean|SHAP|": round(val, 4),
                        })
                top_df = pd.DataFrame(top_data)
                st.dataframe(
                    top_df.pivot(index="Zone", columns="Rank", values="Feature"),
                    use_container_width=True,
                )
        else:
            st.info("Zone-importance data not available for this wave.")


# ============================= TAB 2: SHAP Explorer ========================
with tab_shap:
    st.header("SHAP Explorer")
    st.markdown(
        "Interactive exploration of zone-stratified feature importance. "
        "Toggle between full-feature and no-geo views."
    )

    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        shap_wave = st.selectbox(
            "Wave",
            ALL_WAVES,
            index=3,
            format_func=lambda w: f"Wave {w} ({WAVE_YEARS[w]})",
            key="shap_wave",
        )
    with col_s2:
        geo_toggle = st.radio(
            "Feature set",
            ["All features", "No geography"],
            index=1,
            key="shap_geo",
        )
    no_geo = geo_toggle == "No geography"

    zi = load_zone_importance(shap_wave, no_geo=no_geo)

    if zi is not None and not zi.empty:
        # --- Interactive Heatmap ---
        st.subheader("Zone × Feature Heatmap")
        # Sort columns by overall importance
        col_order = zi.mean(axis=0).sort_values(ascending=False).index
        zi_sorted = zi[col_order]

        fig_hm = px.imshow(
            zi_sorted.round(4),
            color_continuous_scale="RdYlBu_r",
            aspect="auto",
            title=f"Zone-Stratified Feature Importance — Wave {shap_wave} "
                  f"({'no geo' if no_geo else 'all features'})",
            labels={"x": "Feature", "y": "Zone", "color": "mean |SHAP|"},
        )
        fig_hm.update_layout(height=400)
        st.plotly_chart(fig_hm, use_container_width=True)

        # --- Bar chart: overall (zone-averaged) importance ---
        st.subheader("Overall Feature Importance (zone-averaged)")
        overall = zi.mean(axis=0).sort_values(ascending=True)
        fig_bar = px.bar(
            x=overall.values,
            y=overall.index,
            orientation="h",
            title=f"Overall mean |SHAP| — Wave {shap_wave}",
            labels={"x": "mean |SHAP|", "y": "Feature"},
            color=overall.values,
            color_continuous_scale="Viridis",
        )
        fig_bar.update_layout(
            height=max(350, len(overall) * 25),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Per-household strip plot (beeswarm approximation) ---
        st.subheader("Per-Household |SHAP| Distribution (top features)")
        shap_hh = load_shap_by_feature(shap_wave)
        if shap_hh is not None:
            # Determine top N features from the zone-averaged importance
            top_n = st.slider("Number of top features to show", 3, 10, 6, key="shap_topn")
            top_feats = zi.mean(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
            # Filter to features that exist in the per-household data
            top_feats = [f for f in top_feats if f in shap_hh.columns]

            if top_feats:
                melt = shap_hh[["zone"] + top_feats].melt(
                    id_vars="zone", var_name="Feature", value_name="|SHAP|"
                )
                fig_strip = px.strip(
                    melt,
                    x="Feature",
                    y="|SHAP|",
                    color="zone",
                    title=f"Household-level |SHAP| distribution — Wave {shap_wave}",
                    category_orders={"Feature": top_feats},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    stripmode="overlay",
                )
                fig_strip.update_traces(jitter=0.4, marker_size=5, opacity=0.6)
                fig_strip.update_layout(height=450)
                st.plotly_chart(fig_strip, use_container_width=True)
                st.caption(
                    f"Each dot = one of 210 explained households (30/zone × 7 zones). "
                    f"Spread shows within-wave heterogeneity, not cross-seed uncertainty."
                )
        else:
            st.info("Per-household SHAP data not available for this wave.")
    else:
        st.info("Zone-importance data not available for this wave.")


# ============================= TAB 3: Temporal Drift =======================
with tab_drift:
    st.header("Temporal Drift View")
    st.markdown(
        "Five-wave comparison of feature importance (2008/09 → 2020/21). "
        "Highlights the `electricity_source` vs `rural_urban` convergence trend."
    )

    drift_geo = st.radio(
        "Feature set",
        ["All features", "No geography"],
        index=0,
        key="drift_geo",
        horizontal=True,
    )
    drift_no_geo = drift_geo == "No geography"
    comp = load_wave_comparison(drift_no_geo)

    if comp is not None and not comp.empty:
        # --- Slope / line plot ---
        st.subheader("Feature Importance Over Time")

        # Select top N features + always include highlighted ones
        top_n_drift = st.slider("Top features to display", 3, 15, 8, key="drift_topn")
        max_vals = comp.max(axis=1).sort_values(ascending=False)
        top_feats = list(max_vals.head(top_n_drift).index)
        for h in HIGHLIGHT_FEATURES:
            if h in comp.index and h not in top_feats:
                top_feats.append(h)

        comp_top = comp.loc[[f for f in top_feats if f in comp.index]]

        # Reshape for plotly
        plot_data = comp_top.reset_index().melt(
            id_vars=comp_top.index.name or "index",
            var_name="Wave",
            value_name="mean |SHAP|",
        )
        rename_col = comp_top.index.name or "index"
        plot_data = plot_data.rename(columns={rename_col: "Feature"})

        # Mark highlighted features
        plot_data["Highlighted"] = plot_data["Feature"].isin(HIGHLIGHT_FEATURES)

        fig_drift = go.Figure()

        for feat in comp_top.index:
            row = comp_top.loc[feat]
            vals = row.values
            waves = row.index.tolist()
            is_hl = feat in HIGHLIGHT_FEATURES

            fig_drift.add_trace(go.Scatter(
                x=waves,
                y=vals,
                mode="lines+markers",
                name=feat,
                line=dict(
                    width=3.5 if is_hl else 1.5,
                    dash=None if is_hl else None,
                ),
                marker=dict(size=9 if is_hl else 5),
                opacity=1.0 if is_hl else 0.45,
            ))

        # Add within-wave spread as error shading for highlighted features
        for feat in HIGHLIGHT_FEATURES:
            if feat not in comp.index:
                continue
            means = []
            stds = []
            wave_labels = []
            for w in ALL_WAVES:
                hh_data = load_shap_by_feature(w)
                if hh_data is not None and feat in hh_data.columns:
                    means.append(hh_data[feat].mean())
                    stds.append(hh_data[feat].std())
                    wave_labels.append(f"W{w} ({WAVE_YEARS[w]})")
                else:
                    means.append(None)
                    stds.append(None)
                    wave_labels.append(f"W{w} ({WAVE_YEARS[w]})")

            if any(s is not None for s in stds):
                upper = [m + s if m is not None and s is not None else None
                         for m, s in zip(means, stds)]
                lower = [m - s if m is not None and s is not None else None
                         for m, s in zip(means, stds)]

                # Only add band where we have valid data
                valid_waves = [wl for wl, u in zip(wave_labels, upper) if u is not None]
                valid_upper = [u for u in upper if u is not None]
                valid_lower = [l for l in lower if l is not None]

                if valid_waves:
                    fig_drift.add_trace(go.Scatter(
                        x=valid_waves + valid_waves[::-1],
                        y=valid_upper + valid_lower[::-1],
                        fill="toself",
                        fillcolor="rgba(150, 150, 150, 0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{feat} ±1σ",
                    ))

        fig_drift.update_layout(
            title=(
                "SHAP Feature-Importance Drift Across NPS Waves"
                + (" (no geo)" if drift_no_geo else " (all features)")
            ),
            xaxis_title="Wave",
            yaxis_title="Overall mean |SHAP| (zone-averaged)",
            height=520,
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
        )
        st.plotly_chart(fig_drift, use_container_width=True)
        st.caption(
            "**Shaded bands** (for highlighted features) = ±1 standard deviation of "
            "per-household |SHAP| across the 210 explained households in each wave. "
            "This shows within-wave household-level spread, not cross-seed uncertainty "
            "(all waves use a single seed=42). Wider bands mean more heterogeneity "
            "in how much the feature matters across individual households."
        )

        # --- Convergence callout ---
        if not drift_no_geo and "electricity_source" in comp.index and "rural_urban" in comp.index:
            st.subheader("Convergence: electricity_source ≈ rural_urban")
            elec = comp.loc["electricity_source"]
            rural = comp.loc["rural_urban"]
            conv_df = pd.DataFrame({
                "electricity_source": elec,
                "rural_urban": rural,
                "gap": (rural - elec).abs(),
            })
            conv_df.index.name = "Wave"
            st.dataframe(conv_df.round(4), use_container_width=True)
            st.markdown(
                "By Wave 5 (2020/21), `electricity_source` (0.101) has effectively "
                "converged with `rural_urban` (0.101) — a feature that was near-zero "
                "in Wave 1 (0.009) now rivals the urban/rural classification as a "
                "poverty discriminator. "
                "\n\n**Caveat:** electricity missingness drops monotonically across waves "
                "(78% → 75% → 73% → 62% → 30%), tracking the importance rise almost "
                "perfectly. Part of this trend may reflect improving measurement rather "
                "than purely economic change."
            )

        # --- Raw data table ---
        with st.expander("View raw feature × wave table"):
            st.dataframe(comp.round(4), use_container_width=True)
    else:
        st.error("Wave comparison data not found. Run `python -m shap_analysis.compare_waves` first.")
