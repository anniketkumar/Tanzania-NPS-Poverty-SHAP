# Explaining Poverty Clusters Across Space and Time

*A Zone-Stratified SHAP Analysis of Tanzania's National Panel Survey (2008/09–2020/21)*

**Author:** Aniket Kumar Pradhan, NTU Singapore
**Supervisor:** Prof. Snehanshu Saha, APPCAIR, BITS Pilani Goa
**Base paper:** Sende, Saha & Uwimbabazi (2025), *Spatial Distribution of Poverty Clusters and Its Prediction Algorithms*, IEEE Access

---

## What This Is

This project extends a poverty-clustering methodology (K-Means + Stacked Ensemble) with **SHAP explainability** across all **5 NPS waves (2008–2021)** and **7 administrative zones** of Tanzania. It answers: *which household features drive the poverty classification in each zone, and how do those drivers change over 12 years?*

**→ Start here: [Final_Report.md](Final_Report.md)** — the consolidated analysis with all findings and caveats.

**→ Interactive dashboard:** see [Quick Start](#quick-start-dashboard) below.

---

## Quick Start: Dashboard

The dashboard reads pre-computed output files — no heavy ML dependencies needed.

```bash
# Install dashboard dependencies only (streamlit, pandas, plotly, openpyxl)
pip install -r requirements_dashboard.txt

# Launch
streamlit run app.py
```

Three panels:
- **Zone Selector** — pick a wave and zone(s) to view poverty-rate, cluster composition, and top features
- **SHAP Explorer** — interactive heatmap and bar chart of zone-stratified feature importance, with a full-feature / no-geo toggle and per-household strip plots
- **Temporal Drift** — five-wave comparison highlighting the electricity_source vs rural_urban convergence trend

> **Note:** the dashboard requires the pre-computed output files in `baseline_replication/outputs/` and `shap_analysis/outputs/`. If those directories are empty (e.g., a fresh clone with `.gitignore` excluding outputs), run the pipeline first — see [Reproduction](#reproduction) below.

---

## Key Finding

`electricity_source` rose from a near-irrelevant feature (SHAP importance 0.009 in 2008) to the single strongest policy-actionable poverty discriminator (0.101 in 2021) — converging with `rural_urban` over 12 years. Meanwhile, `floor_material` peaked at Wave 3 and declined. As basic housing converges, energy and water access become what separates poor from non-poor.

**Critical caveat:** the ~99% accuracy reflects reconstruction of a K-Means-derived label, not validated poverty prediction. The electrification trend also tracks improving data completeness (78%→30% missingness). See [Final_Report.md §5](Final_Report.md#5-limitations--honest-caveats) for all caveats.

---

## Reports

| Report | Contents |
|---|---|
| [**Final_Report.md**](Final_Report.md) | Consolidated final report — start here |
| [Baseline_Replication_Report.md](Baseline_Replication_Report.md) | Weeks 1–2: pipeline replication + circularity diagnosis |
| [Week3_SHAP_Audit_Report.md](Week3_SHAP_Audit_Report.md) | Week 3: SHAP layer, W4/W5 zone-stratified importance |
| [Week4_Wave3_Extension_Report.md](Week4_Wave3_Extension_Report.md) | Week 4: W3 extension, harmonization, 3-wave drift |
| [Week4_Harmonization_Crosswalk.md](Week4_Harmonization_Crosswalk.md) | Approved value-code crosswalk |
| [Week5_6_Wave1_2_Extension_Report.md](Week5_6_Wave1_2_Extension_Report.md) | Weeks 5–6: W1/W2 extension, 5-wave drift |

---

## Repo Structure

```
tanzania-nps-spatial-analysis/
├── app.py                          # Streamlit dashboard (reads pre-computed outputs)
├── requirements_dashboard.txt      # Dashboard-only dependencies
├── Final_Report.md                 # Consolidated final report
├── baseline_replication/           # Locked baseline pipeline
│   ├── config.py                   #   All tunable parameters
│   ├── run_baseline.py             #   CLI: assemble wave → run pipeline → export
│   ├── leakage_probe.py            #   Random-label control diagnostic
│   ├── data/
│   │   ├── feature_spec.py         #     18 features ↔ raw NPS column mapping
│   │   ├── loader.py               #     Join Sections A/B/C/I
│   │   ├── encode.py               #     One-hot encoding
│   │   ├── harmonize.py            #     Cross-wave value-code harmonization
│   │   └── wave_config.py          #     Per-wave column mappings (W1–W5)
│   ├── pipeline/                   #   KNN → PCA → K-Means → Stacking
│   └── outputs/                    #   experimental_summary, learning curves
├── shap_analysis/                  # SHAP explainability layer
│   ├── config.py                   #   SHAP-specific settings
│   ├── zones.py                    #   Region → zone mapping (7 zones)
│   ├── model.py                    #   Fit pipeline on full wave
│   ├── explain.py                  #   KernelExplainer → zone-stratified matrix
│   ├── plots.py                    #   Heatmaps and bar charts
│   ├── run_shap.py                 #   CLI: fit → explain → save
│   ├── compare_waves.py            #   Cross-wave drift comparison
│   └── outputs/                    #   SHAP matrices, heatmaps, comparisons
├── Week3_SHAP_Audit_Report.md
├── Week4_Wave3_Extension_Report.md
├── Week4_Harmonization_Crosswalk.md
├── Week5_6_Wave1_2_Extension_Report.md
└── Baseline_Replication_Report.md
```

---

## Reproduction

Full pipeline (requires NPS data files in `converted data/wave{N}/household/`):

```bash
# Install all dependencies (ML stack + SHAP + dashboard)
pip install -r baseline_replication/requirements.txt
pip install -r shap_analysis/requirements.txt
pip install -r requirements_dashboard.txt

# 1. Baseline (20-seed evaluation, ~30 min/wave on CPU)
python -m baseline_replication.run_baseline --wave 1  # through --wave 5

# 2. Random-label control (optional, confirms circularity)
python -m baseline_replication.leakage_probe --wave 3

# 3. SHAP analysis (~10 min/wave in sampled mode)
python -m shap_analysis.run_shap --wave 1  # through --wave 5

# 4. Verify SHAP outputs (consistency + additivity)
python -m shap_analysis.verify_outputs

# 5. Cross-wave comparison (seconds)
python -m shap_analysis.compare_waves

# 6. Launch dashboard
streamlit run app.py
```

---

## Environment

Python 3.12, pandas, scikit-learn, PyTorch, shap 0.52, skorch, streamlit. See `requirements.txt` files for pinned versions.
