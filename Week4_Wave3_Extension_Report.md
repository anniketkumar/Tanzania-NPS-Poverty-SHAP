# Week 4 Report — Wave 3 Extension & Three-Wave SHAP Drift Analysis

**Project:** *Explaining Poverty Clusters Across Space and Time — A Zone-Stratified SHAP Analysis of Tanzania's National Panel Survey*
**Replication base:** Sende et al. (2025, IEEE Access)
**Prepared by:** Aniket Kumar Pradhan, NTU Singapore
**Supervisor:** Prof. Snehanshu Saha, APPCAIR, BITS Pilani Goa
**Date:** 14 July 2026
**Scope:** NPS Wave 3 (2012/13), extending the locked Waves 4 (2014/15) and 5 (2020/21); three-wave comparison W3 → W4 → W5

---

## 1. Executive Summary

Week 4 extends the locked baseline + SHAP pipeline back to **NPS Wave 3 (2012/13)** and assembles the first **three-wave feature-importance comparison (2012/13 → 2014/15 → 2020/21)**. The wave-3 data reused the existing `baseline_replication/` and `shap_analysis/` code unchanged in substance — the only new work was **value-code harmonization** (Wave 3's survey uses different codes for a few questions) and a handful of additive diagnostics. Every result was produced by running the pipeline end-to-end, not by trusting cached files.

Headline results:

- **Wave 3 replicates cleanly and behaves like its successors.** A fresh 20-seed run gives **98.78% ± 0.24%** accuracy (AUC 99.95%), essentially identical to Wave 4 (98.80%) and Wave 5 (98.82%). The train/test gap is **+1.18%** — no meaningful over-fitting.
- **The electrification signal is a genuine trend, not a two-point artifact.** `electricity_source` importance rises **monotonically across all three waves — 0.047 → 0.076 → 0.101 — roughly doubling from 2012/13 to 2020/21.** The Wave 4→5 jump I flagged earlier is the tail of a decade-long climb that already starts in Wave 3.
- **The "rural-urban drift" was largely a missing-feature artifact — and is now corrected.** `rural_urban` had been silently dropped from the Wave 5 baseline; recovering it (in W3 and W5) shows the feature is **stably the single strongest driver in every wave (0.099 → 0.116 → 0.101)**, not something that faded between W4 and W5.
- **A coherent convergence story emerges.** `floor_material` — the top actionable driver in Wave 3 (dominant in all 7 zones) — **declines over time (0.093 → 0.082 → 0.053)** as energy, lighting and water access rise. As basic housing quality converges, energy access becomes the differentiator.

The harmonization decisions (with the approved crosswalk), the due-diligence audit, and the caveats — especially a genuine data-quality confound on the electrification trend — are in §3.2, §4 and §6.

---

## 2. What Was Replicated and Extended (Methodology Recap)

**Replicated (locked baseline, unchanged).** The Sende et al. hybrid pipeline, rebuilt in a modular scikit-learn/PyTorch codebase and applied to Wave 3 with the *same* settings as Waves 4/5 (KNN imputation, StandardScaler + PCA(2), K-Means K=2, Stacked Ensemble, 70:30 split, **20 seeds (42–61)**, 5-fold CV inside the stack):

```
raw NPS sections → 18 features → [W3 value-code harmonization] → one-hot
     (126 cols W3 / 133 W4 / 154 W5)
     → KNN imputation → StandardScaler + PCA(2) → K-Means (K=2, poor/non-poor label)
     → Stacked Ensemble (GLM, GBM, RF, XGB, skorch-MLP → logistic meta-learner)
```

The classifier trains on the **raw imputed features**; PCA only manufactures the K-Means label. This is the "no-PCA model" the SHAP layer explains, so attributions map to real survey questions.

**Extended (Week 4).** Three additions, all backward-compatible with the locked Waves 4/5:

1. **Wave 3 harmonization** — a new `baseline_replication/data/harmonize.py` remaps Wave 3's value codes onto the Wave 4/5 reference scheme (§3.2). Waves 4/5 pass through as identity, so their locked results are untouched.
2. **`rural_urban` recovery** — the loader now falls back from `clustertype` (W4-only) to `y{wave}_rural` for W3 and W5, recovering a feature that had been silently dropped from both. W4 still sources `clustertype` and is unchanged.
3. **Additive diagnostics** — internal cluster-validity indices (Silhouette, Calinski-Harabasz, Davies-Bouldin) and a train-vs-test accuracy gap, computed for all three waves; the random-label control extended to Wave 3.

The SHAP layer itself is unchanged: `shap.KernelExplainer` on `P(poor)`, |SHAP| of each one-hot column summed back to its parent feature, then averaged over a stratified **30 households per zone (210 per wave across the 7 zones)** into the zone × feature matrix. Each wave produces a full-feature and a "no-geo" (region + rural_urban removed) variant.

---

## 3. Key Findings

### 3.1 Wave 3 baseline reproduces, and the clusters are only slightly softer

| Metric | Wave 3 (2012/13) | Wave 4 (2014/15) | Wave 5 (2020/21) |
|---|---|---|---|
| Accuracy (test) | 98.78% ± 0.24% | 98.80% ± 0.34% | 98.82% ± 0.25% |
| Train accuracy | 99.96% | 99.96% | 99.99% |
| **Train–test gap** | **+1.18%** | +1.16% | +1.17% |
| AUC (macro) | 99.95% | 99.95% | 99.95% |
| Silhouette | **0.450** | 0.499 | 0.504 |
| Calinski-Harabasz | 3413 | 2818 | 3530 |
| Davies-Bouldin | **0.935** | 0.813 | 0.821 |

Wave 4 and Wave 5 accuracy reproduce their locked figures to within seed noise (Wave 4's SHAP base value later reproduces **bit-identically**, §4), confirming the Week-4 code changes did not disturb them. The train–test gaps (~1.1–1.2%) are small and uniform — the near-perfect accuracy is structural (§6.1), not over-fitting.

**Wave 3's clusters are modestly less separated** than Wave 4/5 — lower Silhouette (0.450 vs ~0.50) and higher Davies-Bouldin (0.935 vs ~0.82). The separation is still solid (a Silhouette of 0.45 is a clearly bimodal split), and accuracy is unaffected, but it is worth noting that Wave 3's poverty partition is marginally fuzzier — a small discount on how sharply the "poor/non-poor" line is drawn in the earliest wave.

### 3.2 Harmonization decisions (the crosswalk actually applied)

Value labels were pulled for **every** feature (not just those pre-flagged) from `master_data_dictionary.csv` and compared cell-by-cell across waves. Only **two** Wave 3 features needed value recodes; everything else matched Wave 4 exactly. Full detail and rationale are in `Week4_Harmonization_Crosswalk.md` (approved 14 Jul 2026).

**Water source (`hh_i19`) — 14-category access taxonomy → 12-category W4/W5 source taxonomy:**

| W3 codes | W3 meaning | → W4/W5 |
|---|---|---|
| 1, 2, 3, 4, **6** | piped inside / standpipes / neighbour / **subsidized vending station** | 1 PIPED WATER |
| 5 | water vendor | 9 CART/SMALL TANK |
| 7 | truck/tanker | 10 TANKER-TRUCK |
| 8 | protected well + pump | 2 TUBEWELL/BOREHOLE |
| 9, 11 | unprotected wells | 4 UNPROTECTED DUGWELL |
| 10 | protected well, no pump | 3 PROTECTED DUG WELL |
| 12 | river/lake/spring/pond | 11 SURFACE WATER |
| 13 → 7 RAINWATER; 14 → 12 OTHER | | |

Code **6 (subsidized vending station) → PIPED WATER** per supervisor decision: a fixed kiosk fed by the piped network is structurally a public standpipe, not mobile delivery. W4/W5's *spring* and *bottled water* categories have no Wave 3 source and are legitimately empty in W3.

**Lighting fuel (`hh_i17`) — codes 9/10 swapped.** Wave 3 uses `9=OTHER, 10=TORCH`; Waves 4/5 use `9=TORCH, 10=OTHER`. This feature was **not** flagged in the original spec — it would have been recoded silently — and Wave 3's codes are swapped onto the W4/W5 positions.

**`rural_urban` canonicalization.** W3's `y3_rural` (0=URBAN, 1=RURAL) is normalized to the W4/W5 convention (1=RURAL, 2=URBAN).

**Three spec renames (all waves — the underlying values were correct, only the names were wrong):** `marital_status` → **`relationship_to_head`** (`hh_b05`), `literacy` → **`grade_currently_attending`** (`hh_c09`), `school_attendance` → **`age_5_or_above`** (`hh_c01`). Applied consistently in code and in every regenerated output (§4).

### 3.3 Zone-stratified importance in Wave 3

Averaged across zones, policy-actionable features only (no-geo), Wave 3's picture is **housing-led**:

| Rank | Wave 3 — mean\|SHAP\| |
|---|---|
| 1 | **floor_material (0.093)** |
| 2 | electricity_source (0.047) |
| 3 | garbage_disposal (0.043) |
| 4 | lighting_fuel (0.038) |
| 5 | toilet_facility (0.036) |

`floor_material` is the **single strongest actionable driver in all 7 zones** in Wave 3 (0.061 in Lake up to 0.133 in Zanzibar), with electricity a consistent second (0.032–0.071, peaking in Zanzibar/Coastal/Northern). Demographics — `head_sex`, `head_age`, `relationship_to_head`, `age_5_or_above`, `grade_currently_attending` — sit at or near 0.00 in every zone, exactly as in Waves 4/5. The zone texture is real: the actionable priority in 2012/13 is overwhelmingly **housing quality (floors)**, only later ceding to energy.

### 3.4 The three-wave drift — is the trend real? (the headline)

Overall (zone-averaged) mean|SHAP|, all three waves:

| Feature | W3 (2012/13) | W4 (2014/15) | W5 (2020/21) | Trajectory |
|---|---|---|---|---|
| **rural_urban** | 0.099 | 0.116 | 0.101 | **stable, always #1 with geo** |
| **electricity_source** | 0.047 | 0.076 | 0.101 | **monotone ↑ (≈ doubles)** |
| floor_material | 0.093 | 0.082 | 0.053 | monotone ↓ |
| lighting_fuel | 0.038 | 0.039 | 0.070 | ↑ (late) |
| water_source | 0.018 | 0.021 | 0.052 | ↑ (late) |
| garbage_disposal | 0.043 | 0.024 | 0.012 | monotone ↓ |
| toilet_facility | 0.036 | 0.031 | 0.046 | ~flat |

Two questions from the Week-3 write-up are now answered:

- **Electrification: a real trend, from Wave 3.** `electricity_source` climbs steadily 0.047 → 0.076 → 0.101 across all three waves. This is **not** a two-point artifact — the 2014/15 → 2020/21 jump is the continuation of a rise that is already underway in 2012/13. Reinforcing it, `lighting_fuel` (0.038 → 0.039 → 0.070) and `water_source` (0.018 → 0.021 → 0.052) also rise, so the broader **energy-and-utilities access** dimension becomes steadily more decisive over the decade, while housing-material features (`floor_material`, `garbage_disposal`) recede.
- **Rural-urban: the earlier "drift" was mostly an artifact.** The Week-3 report noted `rural_urban` as the largest Wave-4 feature but *absent* in Wave 5. That absence was a silent feature drop, not a real disappearance. With the feature recovered in both W3 and W5, `rural_urban` is **stably the top driver in every wave (0.099 → 0.116 → 0.101)** — so the apparent W4→W5 collapse was a measurement gap, now closed.

The coherent story: in 2012/13 the poverty split is defined mainly by **housing quality**; over the decade, as floors converge, **energy and water access take over** as what separates poor from non-poor. Deliverables: `wave_comparison_full.{xlsx,csv,png}` and `wave_comparison_no_geo.{xlsx,csv,png}` (slope plots).

### 3.5 Geographic features (with-geo variant)

With geography included, `rural_urban` is the largest single feature in W3 and W4 and joint-largest in W5, peaking in mixed zones (W3 Coastal 0.195, Northern 0.139). `region` stays small everywhere (0.002–0.064, largest in Coastal — which contains Dar es Salaam — and Zanzibar). As in Week 3, these are reported both with and without geography because their zone-stratified importance is partly tautological (§6.5).

---

## 4. What Was Audited and Verified (Due Diligence)

Every check was performed by executing the pipeline and inspecting fresh outputs.

**We confirmed the ~98.8% is not a label-circularity artifact in Wave 3.** The random-label control — replacing the target with a shuffled label unrelated to the features and re-running the full stack — collapses to **≈ 50% (0.4997 / 0.4917 / 0.5017 across three seeds)**. A proper 5-fold CV that regenerates the K-Means label inside each fold holds at **98.56% ± 0.35%**. So Wave 3 behaves exactly like Wave 4 did in the Week-3 audit: the accuracy comes from the label being a deterministic function of the features, not from leakage.

**We confirmed no leakage in the pipeline.** The KNN imputer, the StandardScaler+PCA, and K-Means are all fit on the **training rows only** and merely transform/assign the test rows (verified in `experiment.py`); the near-perfect train accuracy with a ~1.2% test gap is consistent with a clean split, not memorization across the split.

**We confirmed Wave 4 is genuinely unchanged.** Wave 4's SHAP base value reproduces the locked Week-3 figure **bit-identically (0.403272)**, its design matrix is still 133 one-hot columns, and it still sources `rural_urban` from `clustertype`. The Week-4 changes are backward-compatible.

**We confirmed the renames propagated everywhere.** No output file in `shap_analysis/outputs/` (CSV, Excel, or regenerated plot) contains the old names `marital_status`, `literacy`, or `school_attendance`; all three waves carry the corrected names.

**We confirmed the deliverables are internally consistent and SHAP is sound.** For all three waves: re-aggregating `shap_by_feature_wave{N}.csv` reproduces the `zone_x_feature` Excel to **≤ 8.3e-17**; SHAP local accuracy (`Σ SHAP + E ≈ P(poor)`) holds with **max deviation 3.1e-9 (W3), 8.1e-9 (W4), 5.0e-9 (W5)**; every zone rests on the full **30 households** with the `n_households` column present; and the no-geo files correctly exclude `region`/`rural_urban`. `verify_outputs.py` reports **ALL CHECKS PASSED** for waves 3, 4 and 5.

**One substantive consequence to flag (not a defect):** recovering `rural_urban` in Wave 5 changed its K-Means partition, so **Wave 5's SHAP values now differ from the Week-3 report** (e.g. electricity 0.101 vs the earlier 0.130; base rate `P(poor)` 0.435 vs 0.537). This is the intended effect of adding a strong, previously-missing feature — the Week-4 Wave-5 numbers supersede Week 3's for any cross-wave use. Wave 4 is unaffected.

---

## 5. Reproducibility & Consistency Confirmation

| Check | Result |
|---|---|
| Baseline re-run (20 seeds) | W3 98.78% / W4 98.80% / W5 98.82% — W4/W5 match locked figures |
| Wave 4 SHAP base value | 0.403272 — reproduces the Week-3 value bit-identically |
| Wave 3 random-label control | ≈ 0.50 (0.492–0.502) — no circularity artifact |
| Wave 3 5-fold CV (label refit per fold) | 98.56% ± 0.35% |
| SHAP local accuracy (Σφ + E ≈ f(x)) | max dev 3.1e-9 (W3), 8.1e-9 (W4), 5.0e-9 (W5) |
| Excel matrix ↔ source CSV | agree to ≤ 8.3e-17 (all waves) |
| Per-zone sample size | 30/zone × 7 zones = 210/wave, no shortfalls |
| No-geo variants exclude region/rural_urban | confirmed for all three waves |
| Renames absent from all outputs | confirmed (marital_status / literacy / school_attendance) |

Environment: Python 3.12, pandas, scikit-learn, shap 0.52, skorch. Baseline seeds 42–61 (20); SHAP single seed 42.

---

## 6. Caveats and Points to Discuss

**6.1 Circularity of the K-Means-derived label (inherited).** The poor/non-poor target is generated by K-Means on the same features the classifier predicts, so accuracy measures reconstruction of a self-generated partition, not validated poverty — confirmed again for Wave 3 by the random-label control (§4). The SHAP attributions explain *how the model draws the deprivation split*, which is the right object here, but they remain one step removed from real poverty. Hooking an independent label (Alkire-Foster MPI / a consumption line) into the pipeline's `true_label_col` slot remains the natural non-circular next step.

**6.2 The electrification trend is partly confounded with data completeness — read with care.** `electricity_source` is **structurally missing in 72.6% of Wave 3 households, 61.6% in Wave 4, and ~30% in Wave 5**, and those gaps are filled by KNN imputation. Because the feature becomes progressively *better measured* over exactly the same waves in which its importance rises, **part of the 0.047 → 0.076 → 0.101 climb could reflect improving measurement rather than a purely economic shift.** The monotone trend is real in the model, but I cannot cleanly separate "electricity matters more" from "electricity is measured better" — this is the single most important caveat to raise with the supervisor before the write-up leans on the electrification story.

**6.3 Recovering `rural_urban` re-drew the Wave 5 partition.** As noted in §4, Wave 5's SHAP and base rate changed relative to Week 3 (base `P(poor)` 0.537 → 0.435). This is expected and, in my view, an improvement (W5 was previously missing a strong poverty signal), but any comparison to the Week-3 Wave-5 figures must use the Week-4 numbers.

**6.4 Wave 3 clusters are modestly less separated.** Silhouette 0.450 and Davies-Bouldin 0.935 (vs ~0.50 / ~0.82 in W4/W5) mean the 2012/13 poverty partition is a little fuzzier. Accuracy is unaffected, but the earliest wave's poor/non-poor line is drawn slightly less sharply — a minor discount on Wave-3 interpretation.

**6.5 Water-source harmonization involved judgement.** Mapping Wave 3's 14-category access taxonomy onto the W4/W5 12-category source taxonomy required documented calls (e.g. neighbour/vending-station → piped; well-with-pump → borehole). Feature-level SHAP is category-agnostic (it sums all dummies of a feature), so the three-wave *feature* comparison is robust to these calls; but any future *category-level* water comparison should revisit the crosswalk.

**6.6 Region → zone mapping and Wave 3 coverage (inherited).** The paper publishes no explicit region→zone table; the two ambiguous assignments (Southern Highlands → Southern, Shinyanga → Lake) are unchanged. Wave 3 (2012/13) predates the region splits 22–26, so those codes simply don't appear — all Wave 3 region codes (1–21, 51–55) map cleanly to a zone, so no household was dropped.

**6.7 SHAP is a per-zone sample.** 30 households/zone keeps KernelExplainer tractable; a `--full` run for the publication figures remains an open decision.

---

## 7. Deliverables (in `shap_analysis/outputs/` unless noted)

| File | Contents |
|---|---|
| `Week4_Harmonization_Crosswalk.md` *(repo root)* | The approved value-code crosswalk + flags |
| `experimental_summary_wave{3,4,5}.xlsx` *(baseline_replication/outputs/)* | Per-seed + aggregate metrics incl. train/test gap, Silhouette, CH, DB |
| `zone_importance_wave3.xlsx` / `_no_geo.xlsx` | Wave 3 zone × feature mean\|SHAP\| (+ `n_households`, long form, counts, run config) |
| `zone_importance_wave3_heatmap.png` / `_no_geo_heatmap.png` | Wave 3 heatmaps (with / without geography) |
| `feature_importance_wave3.png` | Wave 3 overall feature-importance bar chart |
| `shap_values_wave3.csv` / `shap_by_feature_wave3.csv` | Wave 3 per-household SHAP (one-hot / aggregated + zone) |
| `wave_comparison_full.{xlsx,csv,png}` | **Three-wave feature × wave table + slope plot (all features)** |
| `wave_comparison_no_geo.{xlsx,csv,png}` | Three-wave comparison without region/rural_urban |

(Waves 4/5 SHAP deliverables were regenerated with the corrected feature names; the Wave-5 files reflect the recovered `rural_urban`.)

**Reproduce:**
```bash
python -m baseline_replication.run_baseline --wave 3   # and --wave 4, --wave 5
python -m baseline_replication.leakage_probe --wave 3  # random-label control
python -m shap_analysis.run_shap --wave 3              # and --wave 4, --wave 5
python -m shap_analysis.verify_outputs                 # consistency + additivity, waves 3/4/5
python -m shap_analysis.compare_waves                  # three-wave comparison table + plots
```

---

*All figures in this report were produced from a clean end-to-end re-run of Waves 3, 4 and 5 on 14 July 2026 and independently verified against the underlying CSVs. Wave 4 reproduces its locked Week-3 figures bit-identically; Wave 5's figures reflect the intended recovery of `rural_urban`. Results are reproducible and internally consistent.*
