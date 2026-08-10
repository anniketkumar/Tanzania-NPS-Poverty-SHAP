# Week 3 Report — Zone-Stratified SHAP Explainability & Due-Diligence Audit

**Project:** *Explaining Poverty Clusters Across Space and Time — A Zone-Stratified SHAP Analysis of Tanzania's National Panel Survey*
**Replication base:** Sende et al. (2025, IEEE Access)
**Prepared by:** Aniket Kumar Pradhan, NTU Singapore
**Supervisor:** Prof. Snehanshu Saha, APPCAIR, BITS Pilani Goa
**Date:** 10 July 2026
**Scope:** NPS Waves 4 (2014/15) and 5 (2020/21)

---

## 1. Executive Summary

Week 3 delivers the SHAP explainability layer proposed for the project: a **zone-stratified feature-importance matrix and heatmap** for both survey waves, built on the locked Week 1–2 baseline. Alongside the deliverable, I ran a full due-diligence pass — re-executing the entire baseline and SHAP pipeline end-to-end and independently verifying each output rather than trusting the files already on disk.

Headline results:

- **The baseline replication reproduces cleanly.** A fresh 20-seed run gives **98.80% ± 0.34%** accuracy on Wave 4 and **98.81% ± 0.22%** on Wave 5 (AUC 99.95% both), matching the Week 1–2 Baseline Report (98.75%) to within seed noise.
- **The zone-stratified SHAP deliverables are internally consistent and now bit-for-bit reproducible.** SHAP local accuracy holds to ~1e-8; Excel matrices match their source CSVs to floating-point precision; two independent SHAP runs of the same wave are now numerically identical.
- **The policy story is stable and interpretable.** Living-standards infrastructure — electricity, lighting fuel, floor material, water source, sanitation — dominates the poverty classification in every zone; household demographics contribute almost nothing.
- **A clear temporal signal emerges between waves:** the weight on **energy access roughly doubles** from Wave 4 to Wave 5, and **water source more than triples**, indicating these dimensions became far more decisive for the poor/non-poor split over the 2014→2021 period.

Every audit item from the prior review was re-checked by running code, not by re-reading the earlier report. All are resolved; the details and the small number of things fixed in the process are in §5.

---

## 2. What Was Replicated and Extended (Methodology Recap)

**Replicated (Weeks 1–2, locked baseline).** The Sende et al. hybrid pipeline, rebuilt in a modular scikit-learn/PyTorch codebase:

```
raw NPS sections → 18 features → one-hot (133 cols W4 / 162 cols W5)
     → KNN imputation → StandardScaler + PCA(2) → K-Means (K=2, poor/non-poor label)
     → Stacked Ensemble (GLM, GBM, RF, XGB, skorch-MLP → logistic meta-learner)
```

The classifier is deliberately trained on the **raw imputed features**, not the PCA coordinates; PCA is used *only* to manufacture the K-Means label. This is the "no-PCA model" the SHAP task requires, so attributions map to real survey questions (floor material, toilet type, …) rather than abstract components.

**Extended (Week 3, SHAP layer).** A thin `shap_analysis/` package that imports the baseline unchanged and adds:

1. **Explain.** `shap.KernelExplainer` on the ensemble's probability of the "poor" cluster, `P(poor)`. KernelExplainer is model-agnostic — required here because the stack contains a neural net and a logistic meta-learner, so the fast TreeExplainer does not apply. Background summarised with k-means (50 rows); a stratified **30 households per zone** are explained (210 per wave across the 7 zones).
2. **Aggregate.** The |SHAP| of every one-hot column is summed back to its parent survey feature (e.g. all `toilet_facility_*` dummies → `toilet_facility`), giving one importance per real feature.
3. **Stratify.** Each household is mapped from its region code to one of the paper's **7 zones** (Western, Lake, Central, Southern, Northern, Coastal, Zanzibar) and averaged → the **zone × feature mean|SHAP|** matrix, the core deliverable.

Each wave produces both a full-feature matrix/heatmap and a **"no-geo" variant** with `region` and `rural_urban` removed, to separate policy-actionable drivers from geographic ones (rationale in §6).

---

## 3. Key Findings

### 3.1 Baseline metrics reproduce the paper and the Week 1–2 report

| Metric | Wave 4 (fresh, 20 seeds) | Wave 5 (fresh, 20 seeds) | Week 1–2 Report | Paper |
|---|---|---|---|---|
| Accuracy | 98.80% ± 0.34% | 98.81% ± 0.22% | 98.75% | 99.1–99.2% |
| F1 (macro) | 98.79% ± 0.35% | 98.81% ± 0.22% | 98.73–98.75% | 98.9–99.1% |
| AUC (macro) | 99.95% ± 0.02% | 99.95% ± 0.02% | 99.94–99.95% | 99.96% |
| MCC | 97.58% ± 0.69% | 97.62% ± 0.44% | 97.47–97.51% | 97.7–98.3% |

The results are within ~0.1 pp of the previously reported figures. As documented in Weeks 1–2, this high accuracy is **structural** — the label is a deterministic function of the features (see caveat §6.1), so the number measures "can the ensemble reconstruct the K-Means partition?", not real-world poverty prediction. That is precisely why the SHAP layer, not the accuracy, carries the analytical weight this week.

### 3.2 Zone-stratified feature importance — what drives the poverty split

Averaged across zones (policy-actionable, geographic features excluded):

| Rank | Wave 4 — mean\|SHAP\| | Wave 5 — mean\|SHAP\| |
|---|---|---|
| 1 | floor_material (0.082) | **electricity_source (0.130)** |
| 2 | electricity_source (0.076) | lighting_fuel (0.097) |
| 3 | lighting_fuel (0.039) | floor_material (0.083) |
| 4 | toilet_facility (0.031) | water_source (0.075) |
| 5 | housing_tenure (0.031) | toilet_facility (0.057) |
| … | garbage_disposal, water_source, education | garbage_disposal, education |

Consistent takeaways in **both** waves:

- **Living-standards infrastructure dominates.** Energy (electricity, lighting fuel), housing quality (floor material), water and sanitation are the top drivers everywhere. This is the direct answer to the proposal's policy question — *"if budget is limited in zone X, prioritise electrification, schooling, or water?"* — and across every zone the answer leans toward **energy and housing/water infrastructure over education or demographics.**
- **Demographics barely matter.** `head_sex`, `head_age`, `marital_status`, `school_attendance` and `literacy` sit at or near 0.00 mean|SHAP| in all zones — the classifier draws the poverty boundary almost entirely from material living conditions.
- **Zone-level texture.** e.g. in Wave 5, `water_source` is strongest in Southern/Western (~0.084–0.090) and `housing_tenure` peaks in Coastal/Northern (~0.057–0.060), so the marginal priority genuinely differs by zone even where the top driver (electricity) is shared.

### 3.3 Temporal drift, Wave 4 → Wave 5

The clearest new finding this week is the **shift in what separates poor from non-poor over time**:

- **Energy access roughly doubles in weight.** electricity_source + lighting_fuel combined rise from ≈0.115 (W4) to ≈0.227 (W5); electricity alone goes from 0.076 to 0.130 and becomes the single dominant driver in every zone.
- **Water source more than triples** (0.021 → 0.075) and **sanitation nearly doubles** (0.031 → 0.057).
- **Floor material stays roughly flat** (0.082 → 0.083), so it *relatively* recedes as energy and water surge.
- The overall base rate of the "poor" cluster also rises (SHAP base value `P(poor)` = 0.403 in W4 vs 0.537 in W5), i.e. a larger share of households falls in the more-deprived cluster in the later wave — interpreted with the circularity caveat (§6.1), this is a shift in the K-Means partition, not a validated poverty-rate change.

Read together: between 2014/15 and 2020/21, **energy and water access became substantially more decisive** for how the model separates deprivation — a coherent, policy-relevant temporal story worth foregrounding in the write-up.

### 3.4 Geographic features (with-geo variant)

In the full-feature matrix, `rural_urban` is the **single largest feature in Wave 4** (zone-avg 0.115), while `region` is small (0.012 in W4, 0.023 in W5). `rural_urban` is absent from Wave 5 (it is not carried in the W5 feature set). The high `rural_urban` weight is expected but has a different interpretation from the actionable drivers (§6.2), which is exactly why the no-geo variant exists.

---

## 4. What Was Audited and Verified (Due Diligence)

Every check below was performed by **executing the pipeline and inspecting fresh outputs**, not by re-reading the prior audit. All eight deliverable files (4 Excel + 4 heatmaps) were regenerated end-to-end for both waves.

**We confirmed the Wave 5 "no-geo" heatmap correctly excludes geography.** The concern was that `zone_importance_wave5_no_geo_heatmap.png` might still be showing a `region` column. On direct inspection of the regenerated image and its backing data, the no-geo heatmap contains **16 feature columns with no `region` or `rural_urban`**, while the with-geo heatmap correctly retains `region`. The plotting path builds the no-geo heatmap from an in-memory geo-dropped matrix (it does not read a cached file), and a full regeneration reproduces the correct, geo-free image. The Wave 4 no-geo heatmap was re-verified the same way and is likewise correct.

**We confirmed every zone rests on the full requested sample.** Each zone in each wave aggregates exactly **30 households** (210 per wave across 7 zones); no zone silently fell below the cap. This count is now surfaced directly in the deliverable as an `n_households` column on the `zone_x_feature` sheet of all four Excel files, so a reader can see the sample backing each row without opening a second sheet.

**We confirmed the deliverables are internally consistent.** Re-aggregating the per-household `shap_by_feature_wave{N}.csv` reproduces the `zone_x_feature` Excel matrix to floating-point precision (max difference ≈ 8e-17). The heatmap cell values match the Excel exactly (spot-checked, e.g. Wave 5 Western: electricity 0.129, lighting 0.113, floor 0.098, water 0.085).

**We confirmed SHAP is mathematically sound (local accuracy).** For all 210 explained households per wave, `Σ SHAP + expected_value ≈ P(poor)` from the model. **Maximum deviation: 8.1e-9 (Wave 4), 4.4e-9 (Wave 5)**; mean deviation ~5e-11. On a random 25-household sample the max deviation is ≤1.8e-11. Additivity effectively holds exactly.

**We confirmed the earlier audit items are resolved:**

- **Factor Analysis:** no FA code exists in any module or pipeline. The dimensionality-reduction stage supports only `pca` and `none`. (`summary_factor_analysis.xlsx` in the project root is a leftover exploratory artifact from earlier reducer comparisons — it is not imported or wired into any code path.)
- **Stale `aniket_code` references:** a full-repository search returns **zero** occurrences; the diagnostic script now imports `baseline_replication` correctly.
- **skorch double-softmax:** the network wrapper no longer applies a manual softmax; with the installed **skorch 1.4.0**, `predict_proba` already normalises internally. Verified empirically — output rows sum to 1.0 with a maximum deviation of **1.2e-7** (floating-point tolerance). No double-softmax.

### What we fixed in the course of the audit

Framed as tightening, not firefighting — nothing here changed the substantive findings:

1. **SHAP reproducibility.** The KernelExplainer's subset sampling and the k-means background were not seeded, so exact SHAP values wobbled by ~0.01 mean|SHAP| run-to-run (rankings and the top-feature set were already stable). We seeded the SHAP computation; two consecutive Wave-4 runs are now **bit-for-bit identical** (max difference 0.0), so the deliverable is fully reproducible.
2. **`n_households` provenance column** added to all four zone-importance workbooks (see above).
3. **Stale baseline summary.** The `experimental_summary.xlsx` on disk had been left from a 2-seed quick run (reporting a trivial 100% on only 120 test rows). Re-running the full 20-seed pipeline replaced it with the correct result (98.80/98.81%, ~1006/1413 test rows), matching the Week 1–2 report.

---

## 5. Reproducibility & Consistency Confirmation

| Check | Result |
|---|---|
| Baseline W4/W5 re-run (20 seeds) | 98.80% / 98.81% accuracy — matches report within seed noise |
| SHAP expected value (base rate) | W4 = 0.4033, W5 = 0.5373 — reproduces bit-identically across runs |
| SHAP local accuracy (Σφ + E ≈ f(x)) | max dev 8.1e-9 (W4), 4.4e-9 (W5) |
| Excel matrix ↔ source CSV | agree to ≤ 8e-17 |
| Heatmap ↔ Excel | values match (spot-checked) |
| No-geo variants exclude region/rural_urban | confirmed for both waves, Excel and heatmap |
| Per-zone sample size | 30/zone × 7 zones = 210 per wave, no shortfalls |
| SHAP determinism (seeded) | two identical runs → max diff 0.0 |

Environment: Python 3.12, pandas 3.0, scikit-learn 1.8, shap 0.52, skorch 1.4. Baseline seeds 42–61 (20); SHAP single seed 42. A standalone checker (`shap_analysis/verify_outputs.py`) reproduces the consistency and additivity results on demand.

---

## 6. Caveats and Points to Discuss

**6.1 Circularity of the K-Means-derived label (methodological, inherited from the paper).** The poor/non-poor target is generated by K-Means on the same features the classifier then predicts. The ~99% accuracy therefore measures reconstruction of a self-generated partition, not real-world poverty prediction — as diagnosed in the Week 1–2 report (random-label control collapses to ~50%). **Implication for SHAP:** the attributions faithfully explain *how the model draws the K-Means boundary*, which is the right object for "which features define the deprivation split," but they are one step removed from validated poverty. The natural next step — hooking an independent poverty label (e.g. Alkire-Foster MPI or a consumption poverty line) into the pipeline's existing `true_label_col` slot — would let us re-run SHAP against a non-circular target. **Worth discussing whether to pursue this before write-up.**

**6.2 Geographic-feature interpretability (region / rural_urban).** These are legitimate features in the paper, but in a *zone-stratified* analysis their SHAP importance is partly tautological ("classified as poor because of where they live"). We therefore report every result **both with and without** geography. The with-geo `rural_urban` importance in Wave 4 (the largest single feature) should be read in this light. Note also that Wave 5 does not carry `rural_urban`, so the two waves are not perfectly symmetric on this axis — a point to flag when comparing across time.

**6.3 Region → zone mapping is an assumption.** The paper does not publish an explicit region→zone table. Ours follows a standard NBS grouping, but two assignments are genuinely ambiguous: the **Southern Highlands** regions are folded into `Southern`, and **Shinyanga** is placed in `Lake`. Both are one-line changes in `zones.py`; **confirming the paper's intended grouping with the authors would remove the last source of ambiguity** in the zone-level comparison.

**6.4 SHAP is a per-zone sample, not the full census.** 30 households/zone (210/wave) keep KernelExplainer tractable and, because zone importance is a mean, give a stable matrix (sampling variation now eliminated by seeding). For the final publication run, `--full` explains every household; worth deciding whether the paper figures should use the full run.

---

## 7. Deliverables (in `shap_analysis/outputs/`)

| File | Contents |
|---|---|
| `zone_importance_wave{4,5}.xlsx` | Zone × feature mean\|SHAP\| (+ `n_households`, long form, per-zone counts, run config) |
| `zone_importance_wave{4,5}_no_geo.xlsx` | Same, with `region`/`rural_urban` removed |
| `zone_importance_wave{4,5}_heatmap.png` | The zone-importance heatmap (with geography) |
| `zone_importance_wave{4,5}_no_geo_heatmap.png` | Heatmap without geographic features |
| `feature_importance_wave{4,5}.png` | Overall (zone-averaged) feature-importance bar chart |
| `shap_values_wave{4,5}.csv` | Per-household signed SHAP over one-hot columns |
| `shap_by_feature_wave{4,5}.csv` | Per-household \|SHAP\| aggregated to 18 features, with zone |

**Reproduce:**
```bash
python -m baseline_replication.run_baseline --wave 4   # and --wave 5
python -m shap_analysis.run_shap --wave 4              # and --wave 5
python -m shap_analysis.verify_outputs                 # consistency + additivity checks
```

---

*All figures in this report were produced from a clean end-to-end re-run of both waves on 10 July 2026 and independently verified against the underlying CSVs. Results are reproducible and internally consistent.*
