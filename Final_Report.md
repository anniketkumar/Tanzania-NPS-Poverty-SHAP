# Final Report — Explaining Poverty Clusters Across Space and Time

*A Zone-Stratified SHAP Analysis of Tanzania's National Panel Survey (2008/09–2020/21)*

**Prepared by:** Aniket Kumar Pradhan, NTU Singapore
**Supervisor:** Prof. Snehanshu Saha, APPCAIR, BITS Pilani Goa
**Date:** 9 August 2026

---

## Abstract

This project extends the Sende et al. (2025, IEEE Access) poverty-clustering methodology — which uses a hybrid K-Means + Stacked Ensemble pipeline to classify Tanzanian households as poor or non-poor — by adding a **SHAP explainability layer** and expanding the analysis from 2 survey waves to all **5 NPS waves (2008/09–2020/21)**. Zone-stratified SHAP analysis reveals *which* household features drive the poverty classification in each of Tanzania's 7 administrative zones, and how those drivers shift over a 12-year period.

The headline finding: **electricity access rose from a near-irrelevant feature in 2008 to the single strongest policy-actionable poverty discriminator by 2021**, effectively converging in importance with the urban/rural classification (SHAP importance 0.009 → 0.101). Meanwhile, housing quality (floor material) — the dominant driver in the early waves — declined over the same period (0.079 → 0.053). This suggests a structural shift: as basic housing converges, energy and water infrastructure become what separates poor from non-poor Tanzanian households.

**Critical caveat:** the ~99% classification accuracy reflects reconstruction of a self-generated K-Means label, not validated poverty prediction. The electrification trend also tracks a monotone improvement in data completeness (78% → 30% missingness), making it impossible to fully separate genuine economic change from improving measurement without external validation.

---

## 1. Motivation & Gap

Sende et al. (2025) demonstrate that a Stacked Ensemble classifier can reconstruct K-Means poverty clusters with >99% accuracy across NPS Waves 4 and 5. Their contribution is methodological — establishing that the hybrid unsupervised→supervised pipeline works — but they do not address **why** the model classifies as it does, nor do they examine how the poverty structure evolves across the full NPS panel.

This project fills two gaps:

1. **Explainability.** SHAP (SHapley Additive exPlanations) is attached to the trained ensemble to produce zone-stratified feature-importance matrices — answering, for each of Tanzania's 7 zones, which features most strongly drive the poverty classification and how they rank against each other.

2. **Temporal depth.** The analysis is extended from the paper's 2 waves (W4, W5) to all 5 NPS waves (W1–W5, spanning 2008/09–2020/21), with documented value-code harmonization, enabling a 12-year view of how the poverty drivers evolve.

The result is an interactive dashboard and the analysis documented in this report, both grounded in the locked, audited pipeline outputs from Weeks 1–6.

---

## 2. Methodology

### 2.1 Baseline Pipeline (Weeks 1–2, locked)

The Sende et al. hybrid pipeline, rebuilt in a modular scikit-learn/PyTorch codebase:

```
raw NPS CSVs → 17–18 features → one-hot (113–154 cols, wave-dependent)
     → KNN imputation → StandardScaler + PCA(2) → K-Means (K=2, poor/non-poor label)
     → Stacked Ensemble (GLM, GBM, RF, XGB, skorch-MLP → logistic meta-learner)
```

- **Classifier trains on raw imputed features**, not PCA coordinates — PCA is used *only* to manufacture the K-Means label. This is essential for SHAP: attributions map to real survey questions, not abstract components.
- **20-seed evaluation** (seeds 42–61), 70:30 train/test split, 5-fold CV inside the stack.
- **No leakage:** all transforms (KNN, Scaler, PCA, K-Means) fit on training data only.

Two tooling substitutions from the original: scikit-learn `StackingClassifier` replaces H2O AutoML; PyTorch MLP via `skorch` replaces H2O Deep Learning.

### 2.2 SHAP Explainability Layer (Week 3)

- **Explainer:** `shap.KernelExplainer` on `P(poor)` — model-agnostic (required because the stack contains an MLP + logistic meta-learner, so TreeExplainer does not apply).
- **Background:** k-means summary of 50 rows; stratified **30 households per zone** (210 per wave) explained.
- **Aggregation:** |SHAP| of each one-hot column summed back to its parent survey feature (e.g., all `toilet_facility_*` dummies → `toilet_facility`).
- **Stratification:** each household mapped from region code to one of the paper's 7 zones via a documented NBS grouping, then averaged → **zone × feature mean|SHAP| matrix**.
- **Dual variant:** every wave produces both a full-feature matrix and a "no-geo" variant (excluding `region` and `rural_urban`), separating policy-actionable drivers from geographic ones.

### 2.3 Five-Wave Harmonization (Weeks 4–6)

Extending the locked W4/W5 pipeline to W1–W3 required value-code harmonization — the survey instruments changed across waves. Key decisions (all documented in `Week4_Harmonization_Crosswalk.md` and the weekly reports):

| Feature | Issue | Resolution |
|---|---|---|
| `water_source` | W1/W3 use a 14-category access taxonomy; W4/W5 use 12-category source taxonomy | Judgement-based crosswalk (e.g., subsidized vending station → piped water per supervisor decision) |
| `lighting_fuel` | Codes 9/10 swapped in W1/W2/W3 vs W4/W5 | Swap applied in harmonization layer |
| `rural_urban` | Different source columns per wave; silently missing in W3/W5 before Week 4 | Recovered via wave-specific fallbacks (`locality`, `y{N}_rural`, `clustertype`); normalized to 1=Rural/2=Urban |
| Column names | W1 uses `scq*`/`sjq*`/`saq*`; W2 uses `hh_j*` for housing | Mapped via `wave_config.py` |
| `child_stool_disposal` | Not collected in W1/W2 | Absent from design matrix; NaN in comparisons |
| Feature renames | 3 spec mislabels (`marital_status`→`relationship_to_head`, etc.) | Corrected in code and all regenerated outputs |

Waves 3–5 pass through unchanged — the harmonization layer is backward-compatible.

*Full audit detail: [Week3_SHAP_Audit_Report.md](Week3_SHAP_Audit_Report.md), [Week4_Wave3_Extension_Report.md](Week4_Wave3_Extension_Report.md), [Week5_6_Wave1_2_Extension_Report.md](Week5_6_Wave1_2_Extension_Report.md).*

---

## 3. Key Findings

### 3.1 Baseline Accuracy Across All Five Waves

| Metric | W1 (2008/09) | W2 (2010/11) | W3 (2012/13) | W4 (2014/15) | W5 (2020/21) |
|---|---|---|---|---|---|
| N (households) | 3,265 | 3,924 | 5,010 | 3,352 | 4,709 |
| Accuracy (test, 20 seeds) | 98.61% ± 0.39% | 98.64% ± 0.32% | 98.78% ± 0.24% | 98.80% ± 0.34% | 98.82% ± 0.25% |
| AUC (macro) | 99.91% | 99.93% | 99.95% | 99.95% | 99.95% |
| Train–test gap | +1.36% | +1.32% | +1.18% | +1.16% | +1.17% |
| Silhouette | 0.476 | 0.469 | 0.450 | 0.499 | 0.504 |
| Base rate P(poor) | 0.164 | 0.268 | 0.257 | 0.403 | 0.435 |

All five waves produce uniformly >98.5% accuracy and >99.9% AUC. The accuracy is structural — see §5.1.

### 3.2 The Five-Wave Feature-Importance Trajectory

Overall (zone-averaged) mean|SHAP|, all five waves (no-geo):

| Feature | W1 (2008/09) | W2 (2010/11) | W3 (2012/13) | W4 (2014/15) | W5 (2020/21) | Trajectory |
|---|---|---|---|---|---|---|
| **electricity_source** | **0.009** | **0.017** | **0.047** | **0.076** | **0.101** | **monotone ↑ — 11× in 12 years** |
| floor_material | 0.079 | 0.090 | 0.093 | 0.082 | 0.053 | ↑ early, then monotone ↓ |
| lighting_fuel | 0.030 | 0.070 | 0.038 | 0.039 | 0.070 | non-monotone |
| water_source | 0.016 | 0.020 | 0.018 | 0.021 | 0.052 | flat then late ↑ |
| toilet_facility | 0.010 | 0.048 | 0.036 | 0.031 | 0.046 | jump W1→W2, then ~flat |
| garbage_disposal | 0.039 | 0.016 | 0.043 | 0.024 | 0.012 | long-run ↓ |
| housing_tenure | 0.036 | 0.042 | 0.033 | 0.031 | 0.026 | gradual ↓ |

With geography included:

| Feature | W1 | W2 | W3 | W4 | W5 |
|---|---|---|---|---|---|
| **rural_urban** | 0.048 | 0.068 | 0.099 | 0.116 | 0.101 |
| **electricity_source** | 0.009 | 0.017 | 0.047 | 0.076 | 0.101 |

### 3.3 Headline: The Electrification / Urbanicity Convergence

The project's central finding is the **convergence of `electricity_source` with `rural_urban`** as a poverty discriminator:

```
electricity_source:  0.009 → 0.017 → 0.047 → 0.076 → 0.101
rural_urban:         0.048 → 0.068 → 0.099 → 0.116 → 0.101
```

By Wave 5, the two features are effectively tied at 0.101. A feature that was near-irrelevant in 2008 now rivals the urban/rural classification as a poverty discriminator.

Simultaneously, `floor_material` — the top actionable driver in all 7 zones in Waves 1–3 — peaked at Wave 3 (0.093) and then declined monotonically (0.082 → 0.053). The coherent story: **as basic housing quality converges across the population, energy and water access become what separates poor from non-poor.**

### 3.4 Zone-Level Texture

The temporal story is not uniform across zones. Zone-stratified analysis reveals:

- **Wave 1 (2008/09):** `floor_material` is the #1 actionable driver in all 7 zones, with `housing_tenure` and `garbage_disposal` as secondary drivers. `electricity_source` is near-zero everywhere.
- **Wave 3 (2012/13):** `floor_material` still dominates all zones, but `electricity_source` has risen to a consistent second (0.032–0.071, peaking in Zanzibar/Coastal/Northern).
- **Wave 5 (2020/21):** `electricity_source` leads or co-leads in every zone. `water_source` is strongest in Southern/Western (~0.084–0.090), while `housing_tenure` peaks in Coastal/Northern — the marginal policy priority genuinely differs by zone.
- **Demographics barely matter in any wave.** `head_sex`, `head_age`, `relationship_to_head`, `age_5_or_above`, and `grade_currently_attending` sit at or near 0.00 mean|SHAP| in all zones across all waves.

### 3.5 Complications in the Energy Narrative

The energy-access story is not entirely clean:

1. **`lighting_fuel` is non-monotone** (0.030 → 0.070 → 0.038 → 0.039 → 0.070). The Wave 2 spike and Wave 3 drop are anomalous. This may reflect harmonization sensitivity (the code 9/10 swap) or genuine technology transition volatility.
2. **The rural_urban "drift" was partly a missing-feature artifact.** The original Week 3 analysis showed `rural_urban` disappearing in Wave 5. This was a silent feature drop, corrected in Week 4 — `rural_urban` is stably #1 with geography in every wave once recovered.

---

## 4. Due Diligence Summary

Rigorous auditing was a core principle throughout the project. This section condenses the verification work from Weeks 3–6 — full details are in the individual weekly reports.

### 4.1 Leakage & Circularity Checks

| Check | Waves Verified | Result |
|---|---|---|
| Random-label control (shuffled target) | W3, W4 | Accuracy collapses to ~50% (chance), confirming the ~99% is structural |
| Proper 5-fold CV (K-Means refit per fold) | W3, W4 | 98.56–98.9% — holds because label IS a function of features |
| Train–test gap | All 5 | 1.16–1.36% — no meaningful overfitting |
| Pipeline leakage (KNN/Scaler/PCA/K-Means fit-transform boundary) | All 5 | Transforms fit on training data only, confirmed by code inspection |

> **Open item:** the random-label control was NOT run for Waves 1 and 2 specifically. The mechanism is structural and wave-independent, so the result is expected to hold, but it is strictly unverified.

### 4.2 SHAP Validity

| Check | Result |
|---|---|
| Local accuracy: Σφ + E[f(x)] ≈ f(x) | Max deviation across all 5 waves: 1.99e-8 (W2). Typical: ~5e-10 |
| Re-aggregation: per-household CSV ↔ zone Excel | Max discrepancy ≤ 8.33e-17 (floating-point noise) |
| No-geo variants correctly exclude region/rural_urban | Confirmed for all 5 waves, both Excel and heatmap |
| Per-zone sample size | 30/zone × 7 zones = 210/wave in every wave, no shortfalls |
| SHAP determinism (seeded) | Two identical runs → max difference 0.0 |

### 4.3 Self-Caught Errors and Fixes

Transparency requires documenting not just what passed, but what was found and corrected:

1. **SHAP reproducibility (Week 3).** KernelExplainer's subset sampling and k-means background were initially unseeded, causing ~0.01 mean|SHAP| wobble between runs. Fixed by seeding the NumPy RNG → bit-identical runs.
2. **`rural_urban` silent feature drop (Week 4).** The loader sourced `rural_urban` only from `clustertype` (W4-only), silently dropping it from W3 and W5. Recovered via wave-specific fallbacks. This changed Wave 5's SHAP values (e.g., base P(poor) 0.537→0.435) and corrected the apparent "rural_urban disappears in W5" finding.
3. **Feature renames (Week 4).** Three features were mislabeled in the spec: `marital_status`→`relationship_to_head`, `literacy`→`grade_currently_attending`, `school_attendance`→`age_5_or_above`. Corrected in code and all regenerated outputs.
4. **Stale experimental summary (Week 3).** A cached 2-seed Excel was replaced by the correct 20-seed run.

### 4.4 Backward Compatibility

Every extension (W3 in Week 4, W1/W2 in Weeks 5–6) was verified to leave locked wave results unchanged:

- Wave 4 SHAP base value (0.403272) reproduces bit-identically across all weeks.
- All W3/W4/W5 accuracy, AUC, Silhouette, and SHAP values reproduce to within rounding.

---

## 5. Limitations & Honest Caveats

### 5.1 Label Circularity (critical — inherited from the paper)

The poor/non-poor target is generated by K-Means on the same features the classifier then predicts. The ~99% accuracy measures "can the ensemble reconstruct the K-Means partition?", not real-world poverty prediction. This is confirmed by the random-label control collapsing to ~50%.

**Implication for SHAP:** the attributions explain *how the model draws the K-Means boundary*, which is the right object for "which features define the deprivation split." But they are one step removed from validated poverty. The substantially varying base rates across waves — P(poor) = 0.164 (W1) to 0.435 (W5) — further illustrate that the K-Means partition is a model construct, not a validated poverty line.

### 5.2 Electrification / Missingness Confound (critical)

The five-wave missingness gradient for `electricity_source`:

```
78.2% → 75.1% → 72.6% → 61.6% → 29.9%
```

tracks the importance gradient almost perfectly:

```
0.009 → 0.017 → 0.047 → 0.076 → 0.101
```

Both are monotone. Both span an order of magnitude. **It is impossible from these data alone to determine how much of the importance rise reflects genuine economic change versus improving survey measurement.** The trend is real in the model; whether it reflects Tanzania's actual electrification-led poverty restructuring, or improving data coverage (or both), cannot be separated without external validation (e.g., TANESCO grid extension records, REA data).

### 5.3 Judgement-Based Crosswalk Decisions

The water-source harmonization maps Wave 1/3's 14-category access taxonomy onto W4/W5's 12-category source taxonomy via documented but subjective calls (e.g., "subsidized vending station → piped water," "well with pump → borehole"). Feature-level SHAP sums all dummies, so the feature-level comparison is robust to these calls, but any future category-level analysis should revisit the crosswalk.

The `lighting_fuel` code 9/10 swap (W1/W2/W3 vs W4/W5) affects KNN imputation values, potentially contributing to the non-monotone `lighting_fuel` trajectory.

### 5.4 Small SHAP Sample Size

30 households per zone (210 per wave) explained via KernelExplainer. The means are stable (confirmed by bit-reproducibility), but the sample is thin for capturing within-zone heterogeneity, especially in Central zone (144 total HH in W1, of which 30 are explained — a 21% subsample).

### 5.5 `lighting_fuel` Volatility

The non-monotone trajectory (0.030 → 0.070 → 0.038 → 0.039 → 0.070) is anomalous relative to other features' smoother paths. It may reflect genuine rapid technology change, harmonization sensitivity, or KNN imputation interacting with changing missing-data patterns.

### 5.6 Region → Zone Mapping Assumption

The paper does not publish an explicit region→zone table. The mapping follows a standard NBS grouping, but two assignments are ambiguous: **Southern Highlands** regions folded into Southern, and **Shinyanga** placed in Lake (vs. the older Western grouping). Confirming with the paper's authors would remove this source of ambiguity.

### 5.7 W1 Rural-Urban Code 3 = "Mixture"

Wave 1's locality variable includes a third category ("Mixture") not present in other waves, mapped to Urban. Alternative mappings were not tested. Since `rural_urban` is the strongest feature in the with-geo model, this has non-trivial SHAP consequences for affected households.

### 5.8 Random-Label Control Not Run for W1/W2

The structural argument (K-Means label ⇒ deterministic function of features ⇒ control collapses to ~50%) almost certainly applies, but this is unverified for Waves 1 and 2 specifically.

---

## 6. Path Forward

Per discussions with supervisor, the following would be needed before this work is publication-ready:

### 6.1 External Validation (essential)

The single most important next step is hooking an **independent poverty measure** — Alkire-Foster MPI, a consumption-based poverty line, or similar — into the pipeline's `true_label_col` slot. This would:
- Replace the circular K-Means label with a non-tautological target
- Enable genuine poverty-prediction accuracy measurement
- Allow SHAP to attribute features against *real* poverty, not the model's own partition

### 6.2 Uncertainty Quantification (essential)

The current trend line is a single-seed point estimate. Publication-quality results should include:
- **Multi-seed SHAP runs** (varying both the model seed and the SHAP background/sample) to produce confidence intervals on the trend
- **Bootstrap CI** on the zone-averaged importance values
- The dashboard's within-wave household spread is a start, but is not a substitute for proper uncertainty propagation

### 6.3 Related Work Section (essential)

A proper related-work survey covering SHAP-based poverty analysis, feature-importance drift in panel data, and the broader literature on ML-based poverty mapping (e.g., Jean et al. 2016 satellite imagery, Blumenstock et al. 2015 mobile phone data) would contextualize this contribution.

### 6.4 Full SHAP Run

The `--full` flag explains every household rather than a 30/zone sample. Publication figures should use the full run to eliminate any sampling bias.

### 6.5 Robustness Checks

- Test alternative region→zone assignments (Shinyanga in Western; Southern Highlands as separate zone)
- Test alternative `rural_urban` mapping for W1 code 3 ("Mixture" → Rural instead of Urban)
- Test sensitivity of the `water_source` crosswalk by comparing the coarser common-taxonomy alternative

---

## 7. Deliverables

### 7.1 Code & Data

| Deliverable | Location |
|---|---|
| Baseline pipeline | [`baseline_replication/`](baseline_replication/) |
| SHAP analysis pipeline | [`shap_analysis/`](shap_analysis/) |
| Interactive dashboard | [`app.py`](app.py) — launch with `streamlit run app.py` |
| Dashboard dependencies | [`requirements_dashboard.txt`](requirements_dashboard.txt) |

### 7.2 Output Files (pre-computed, in `shap_analysis/outputs/`)

| File pattern | Contents |
|---|---|
| `zone_importance_wave{1–5}.xlsx` / `_no_geo.xlsx` | Zone × feature mean\|SHAP\| matrices (+ n_households, long form, counts, run config) |
| `zone_importance_wave{1–5}_heatmap.png` / `_no_geo_heatmap.png` | Zone-importance heatmaps |
| `feature_importance_wave{1–5}.png` | Overall feature-importance bar charts |
| `shap_values_wave{1–5}.csv` | Per-household signed SHAP over one-hot columns |
| `shap_by_feature_wave{1–5}.csv` | Per-household \|SHAP\| aggregated to features + zone |
| `wave_comparison_full.{xlsx,csv,png}` | Five-wave feature × wave table + slope plot (all features) |
| `wave_comparison_no_geo.{xlsx,csv,png}` | Five-wave comparison without geography |
| `experimental_summary_wave{1–5}.xlsx` | Per-seed + aggregate baseline metrics |
| `learning_curve_wave{1–5}.png` | Learning curves |

### 7.3 Reports (in repo root)

| Report | Covers |
|---|---|
| [Baseline_Replication_Report.md](Baseline_Replication_Report.md) | Weeks 1–2: pipeline replication, circularity diagnosis |
| [Week3_SHAP_Audit_Report.md](Week3_SHAP_Audit_Report.md) | Week 3: SHAP layer, W4/W5 zone-stratified importance, full audit |
| [Week4_Wave3_Extension_Report.md](Week4_Wave3_Extension_Report.md) | Week 4: W3 extension, harmonization, 3-wave drift, rural_urban correction |
| [Week4_Harmonization_Crosswalk.md](Week4_Harmonization_Crosswalk.md) | Approved value-code crosswalk for W3 |
| [Week5_6_Wave1_2_Extension_Report.md](Week5_6_Wave1_2_Extension_Report.md) | Weeks 5–6: W1/W2 extension, 5-wave drift, missingness confound |
| [Final_Report.md](Final_Report.md) | This document — consolidated final report |

---

*This report consolidates findings from Weeks 1–6 of the project. All figures derive from the locked, audited pipeline outputs. The weekly reports contain the full audit trail; this document synthesizes the results into a coherent narrative with all caveats carried forward.*
