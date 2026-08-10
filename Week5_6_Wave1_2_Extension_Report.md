# Week 5–6 Report — Waves 1 & 2 Extension & Five-Wave SHAP Drift Analysis

**Project:** *Explaining Poverty Clusters Across Space and Time — A Zone-Stratified SHAP Analysis of Tanzania's National Panel Survey*
**Replication base:** Sende et al. (2025, IEEE Access)
**Prepared by:** Aniket Kumar Pradhan, NTU Singapore
**Supervisor:** Prof. Snehanshu Saha, APPCAIR, BITS Pilani Goa
**Date:** 31 July 2026
**Scope:** NPS Wave 1 (2008/09) and Wave 2 (2010/11), extending the locked Waves 3–5; five-wave comparison W1 → W2 → W3 → W4 → W5

---

## 1. Executive Summary

Weeks 5–6 extend the locked pipeline back to **NPS Waves 1 (2008/09) and 2 (2010/11)**, completing the full **five-wave panel (2008–2021)**. The harmonization crosswalk approved in the prior conversation was applied; no code changes touched the locked Waves 3–5.

Headline findings:

- **Waves 1 and 2 replicate cleanly.** 20-seed runs give **98.61% ± 0.39%** (W1) and **98.64% ± 0.32%** (W2) accuracy, consistent with the 98.78–98.82% range of the locked W3–5. AUC > 99.9% in both. Train–test gaps (1.36% W1, 1.32% W2) are small and in the same band as later waves.

- **The "electrification converges with rural_urban" story from Week 4 is strengthened by the two earlier data points.** `electricity_source` importance was near-zero in W1 (0.009) and W2 (0.017), then rose monotonically to 0.047 → 0.076 → 0.101 across W3–W5. The five-wave trajectory shows the climb beginning from a genuinely low base, not just emerging between W3 and W4.

- **However, the electrification/missingness confound is now MORE serious, not less.** Electricity missingness is **78.2% in W1 and 75.1% in W2** — worse than Wave 3's already-flagged 72.6%. The five-wave missingness gradient (78% → 75% → 73% → 62% → 30%) is almost perfectly monotone, making it harder — not easier — to separate "electricity matters more" from "electricity is measured better."

- **`floor_material` was already the top actionable driver in 2008 and remains so through 2012/13**, only fading from Wave 4 onward (0.079 → 0.090 → 0.093 → 0.082 → 0.053). The housing-to-energy shift story is confirmed and tightened: housing dominated the poverty partition throughout the first half of the panel, and energy access overtakes it in the second half.

- **W1/W2 cluster separation is modestly softer** (Silhouette 0.476/0.469) than W4/W5 (~0.50), consistent with Wave 3's pattern. The poor/non-poor line is drawn slightly less sharply in earlier waves but accuracy is unaffected.

- **One important gap: the random-label control was NOT run for W1/W2.** The `leakage_probe.py` module exists and was confirmed for W3 and earlier for W4, but I found no evidence it was executed for the two new waves. The structural argument from W3/W4 (K-Means label ⇒ deterministic, ∴ collapse to ~50% expected) almost certainly applies, but this is an open verification item.

- **All W3/W4/W5 locked figures are genuinely unchanged.** Every metric reproduces to within rounding of the Week 4 report's locked values. The addition of W1/W2 did not alter any upstream results.

---

## 2. Methodology Recap

**Replicated (locked baseline, unchanged).** The Sende et al. hybrid pipeline applied to Waves 1 and 2 with identical settings to W3–5:

```
raw NPS sections → 17 features (W1/W2; 18 for W3–5 which add child_stool_disposal)
     → [W1/W2 harmonization] → one-hot
     (113 cols W1 / 115 W2 / 126 W3 / 133 W4 / 154 W5)
     → KNN imputation → StandardScaler + PCA(2) → K-Means (K=2)
     → Stacked Ensemble (GLM, GBM, RF, XGB, skorch-MLP → logistic meta-learner)
```

20 seeds (42–61), 70:30 train/test split, 5-fold CV inside the stack.

**Extended (Weeks 5–6).** Two additions, backward-compatible:

1. **Wave 1/2 harmonization** — `wave_config.py` maps W1/W2 column names (which differ substantially from W3–5: e.g. `scq6` → `hh_c06`, `sjq*` → `hh_j*`) onto the canonical scheme; `harmonize.py` remaps value codes. Waves 3–5 pass through unchanged.
2. **`child_stool_disposal` absence** — this feature (`hh_i15`) does not exist in W1/W2 survey instruments. The pipeline correctly handles this: the feature column is absent from the design matrix and appears as NaN in the wave-comparison tables. No imputation or placeholder was applied — the feature genuinely didn't exist before Wave 3.

**Harmonization decisions applied for W1/W2** (from the approved crosswalk, conversation `8317db06`):

| Feature | W1/W2 Issue | Resolution |
|---|---|---|
| **Column names** | W1 uses `scq*`, `sjq*`, `saq*` prefixes; W2 uses `hh_b*`, `hh_c*`, `hh_j*` | Mapped via `wave_config.py` column_map |
| **`rural_urban`** | W1: `locality` (1=Rural, 2=Urban, 3=Mixture); W2: `y2_rural` (0=Urban, 1=Rural) | Normalized to W4/W5 convention (1=Rural, 2=Urban); W1 code 3 (Mixture) → 2 (Urban) |
| **`water_source`** | W1: `sjq19` with 14-category access taxonomy (same as W3) | Same crosswalk as W3 applied |
| **`lighting_fuel`** | W1/W2 code 9/10 may be swapped (same issue as W3) | Same swap as W3 applied |
| **`electricity_source`** | W2: `hh_j16` uses codes matching W4/W5 (no remap needed); W1: `sjq16` same taxonomy | Identity mapping |
| **`child_stool_disposal`** | Not collected in W1/W2 | Absent from design matrix; NaN in comparisons |

---

## 3. Key Findings

### 3.1 Baseline metrics across all five waves

| Metric | W1 (2008/09) | W2 (2010/11) | W3 (2012/13) | W4 (2014/15) | W5 (2020/21) |
|---|---|---|---|---|---|
| N (households) | 3,265 | 3,924 | 5,010 | 3,352 | 4,709 |
| One-hot columns | 113 | 115 | 126 | 133 | 154 |
| Accuracy (test) | **98.61% ± 0.39%** | **98.64% ± 0.32%** | 98.78% ± 0.24% | 98.80% ± 0.34% | 98.82% ± 0.25% |
| Train accuracy | 99.97% | 99.97% | 99.96% | 99.96% | 99.99% |
| **Train–test gap** | **+1.36%** | **+1.32%** | +1.18% | +1.16% | +1.17% |
| AUC (macro) | 99.91% | 99.93% | 99.95% | 99.95% | 99.95% |
| Silhouette | **0.476** | **0.469** | 0.450 | 0.499 | 0.504 |
| Calinski-Harabasz | **2,000** | **2,592** | 3,413 | 2,818 | 3,530 |
| Davies-Bouldin | **0.943** | **0.897** | 0.935 | 0.813 | 0.821 |
| Base rate P(poor) | 0.164 | 0.268 | 0.257 | 0.403 | 0.435 |

**All five waves produce essentially the same pipeline quality.** Accuracy is uniformly > 98.5%, AUC > 99.9%, and train–test gaps are small (1.1–1.4%). The slight deterioration in W1/W2 accuracy and gap is within normal cross-wave variation and does not indicate a pipeline problem.

**Cluster quality in W1/W2.** Silhouette scores (0.476 W1, 0.469 W2) are comparable to or better than W3 (0.450), though below W4/W5 (~0.50). Davies-Bouldin is highest in W1 (0.943), meaning the clusters overlap slightly more. The earliest waves' poverty partitions are a little fuzzier, but still clearly bimodal.

**Notably, the base rate P(poor) varies substantially across waves** — from 0.164 in W1 to 0.435 in W5. This reflects the K-Means partition shifting as feature distributions change, not a real poverty trend. It is a reminder that the "poor/non-poor" label is a model construct, not a validated poverty line.

### 3.2 Zone-stratified results for W1/W2

**Zone composition.** Both waves use the same 7-zone mapping as W3–5 (Western, Lake, Central, Southern, Northern, Coastal, Zanzibar). All 26 W1/W2 region codes (1–21, 51–55) map cleanly to zones; no post-2012 region splits (codes 22–26) exist in these early waves.

**Zone sample sizes:**

| Zone | W1 n_total | W2 n_total | W3–5 range |
|---|---|---|---|
| Western | 216 | 279 | 330–450 |
| Lake | 440 | 604 | 700–800 |
| **Central** | **144** | **178** | 250–350 |
| Southern | 855 | 1,013 | 1,200–1,800 |
| Northern | 400 | 445 | 530–660 |
| Coastal | 731 | 869 | 900–1,250 |
| Zanzibar | 479 | 536 | 600–700 |
| **Total** | **3,265** | **3,924** | 5,010 / 3,352 / 4,709 |

> [!WARNING]
> **Central zone is thin in W1 (144 households) and W2 (178).** The SHAP analysis explains 30 households/zone, so the Central zone's 30 are drawn from a much smaller pool than other zones. While 144 is still viable for KernelExplainer, any Central-specific finding in W1 should be interpreted with more caution than, say, Southern (855).

**Top features by zone (no-geo, Wave 1):**

| Zone | #1 | #2 | #3 |
|---|---|---|---|
| Western | floor_material (0.048) | housing_tenure (0.031) | garbage_disposal (0.026) |
| Lake | floor_material (0.076) | housing_tenure (0.043) | garbage_disposal (0.019) |
| Central | floor_material (0.055) | housing_tenure (0.020) | garbage_disposal (0.022) |
| Southern | floor_material (0.064) | housing_tenure (0.026) | garbage_disposal (0.028) |
| Northern | floor_material (0.095) | housing_tenure (0.046) | lighting_fuel (0.043) |
| Coastal | floor_material (0.115) | garbage_disposal (0.102) | housing_tenure (0.062) |
| Zanzibar | floor_material (0.102) | garbage_disposal (0.072) | lighting_fuel (0.045) |

**`floor_material` is the single strongest actionable driver in all 7 zones in Wave 1** — exactly as it was in Wave 3, confirming the housing-dominance story extends back to 2008.

**Top features by zone (no-geo, Wave 2):**

| Zone | #1 | #2 | #3 |
|---|---|---|---|
| Western | floor_material (0.077) | lighting_fuel (0.054) | toilet_facility (0.045) |
| Lake | floor_material (0.065) | lighting_fuel (0.051) | toilet_facility (0.037) |
| Central | floor_material (0.066) | lighting_fuel (0.050) | toilet_facility (0.042) |
| Southern | floor_material (0.078) | lighting_fuel (0.055) | toilet_facility (0.048) |
| Northern | floor_material (0.089) | lighting_fuel (0.072) | toilet_facility (0.045) |
| Coastal | floor_material (0.126) | lighting_fuel (0.104) | housing_tenure (0.088) |
| Zanzibar | floor_material (0.132) | lighting_fuel (0.102) | toilet_facility (0.074) |

Wave 2 shows `lighting_fuel` emerging as a strong second driver (it was less prominent in W1), while `garbage_disposal` recedes. `floor_material` remains #1 in every zone.

### 3.3 The five-wave drift — the full trajectory

Overall (zone-averaged) mean|SHAP|, all five waves:

| Feature | W1 (2008/09) | W2 (2010/11) | W3 (2012/13) | W4 (2014/15) | W5 (2020/21) | Trajectory |
|---|---|---|---|---|---|---|
| **rural_urban** | 0.048 | 0.068 | 0.099 | 0.116 | 0.101 | **↑ then plateau (always #1 with geo from W3 on)** |
| **electricity_source** | **0.009** | **0.017** | 0.047 | 0.076 | 0.101 | **monotone ↑ — 11× increase over 12 years** |
| floor_material | 0.079 | 0.090 | 0.093 | 0.082 | 0.053 | **↑ early, then monotone ↓** |
| lighting_fuel | 0.030 | **0.070** | 0.038 | 0.039 | 0.070 | **non-monotone (high W2, dip W3–4, recover W5)** |
| water_source | 0.016 | 0.020 | 0.018 | 0.021 | 0.052 | **flat then late ↑** |
| toilet_facility | 0.010 | 0.048 | 0.036 | 0.031 | 0.046 | **jump W1→W2, then ~flat** |
| garbage_disposal | 0.039 | 0.016 | 0.043 | 0.024 | 0.012 | **non-monotone; long-run ↓** |
| housing_tenure | 0.036 | 0.042 | 0.033 | 0.031 | 0.026 | **gradual ↓** |
| region | 0.018 | 0.018 | 0.017 | 0.012 | 0.011 | **gradual ↓** |
| education_level | 0.009 | 0.012 | 0.013 | 0.017 | 0.018 | **gradual ↑** |
| head_age | 0.014 | 0.014 | 0.007 | 0.002 | 0.003 | **↓ (negligible by W4)** |
| grade_currently_attending | 0.016 | 0.009 | 0.005 | 0.008 | 0.009 | **↓ early, flat** |
| child_stool_disposal | — | — | 0.009 | 0.006 | 0.007 | **W3–5 only; small** |
| rooms | 0.001 | 0.001 | 0.014 | 0.008 | 0.005 | **spike in W3, then ↓** |
| household_size | 0.001 | 0.001 | 0.002 | 0.003 | 0.004 | **gradual ↑, tiny** |
| head_sex | 0.001 | 0.002 | 0.001 | 0.001 | 0.001 | **~zero, stable** |
| relationship_to_head | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | **~zero** |
| age_5_or_above | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **zero** |

### 3.4 Does the "electrification converges with rural_urban" story hold up?

**Yes — it strengthens.** The Week 4 report identified a monotone rise in `electricity_source` across W3→W4→W5 (0.047 → 0.076 → 0.101). Adding W1 and W2 reveals the feature was near-zero in 2008 (0.009) and still small in 2010 (0.017). The full trajectory is:

```
electricity_source:  0.009 → 0.017 → 0.047 → 0.076 → 0.101
rural_urban:         0.048 → 0.068 → 0.099 → 0.116 → 0.101
```

By Wave 5, `electricity_source` has effectively converged with `rural_urban` in importance (0.101 vs 0.101). The convergence is real in the model: what was once a near-irrelevant feature now rivals the urban/rural classification as a poverty discriminator.

**But `lighting_fuel` complicates the energy-access narrative.** Unlike `electricity_source`'s clean monotone rise, `lighting_fuel` shows a non-monotone path: 0.030 → 0.070 → 0.038 → 0.039 → 0.070. The W2 spike (0.070) is anomalously high relative to W1 and W3. This may reflect harmonization sensitivity (the W1/W2 lighting code swap has a direct counterpart), or it may be genuine volatility in how lighting technology maps to poverty. Either way, the "energy access steadily replaces housing" story is cleaner for electricity than for lighting.

**`floor_material` peaked in W3, not W1.** The five-wave trajectory (0.079 → 0.090 → 0.093 → 0.082 → 0.053) shows floor material importance *rising* from W1 to W3, then declining. This is a modest revision to the Week 4 narrative, which framed floor_material as simply declining over time. The peak is in 2012/13, and the decline begins in 2014/15 — roughly when electrification starts rising fast.

### 3.5 Electricity missingness — the confound gets worse

| Wave | electricity_source missingness | Note |
|---|---|---|
| W1 (2008/09) | **78.2%** (2,553/3,265) | Worst in panel |
| W2 (2010/11) | **75.1%** (2,948/3,924) | |
| W3 (2012/13) | 72.6% (3,635/5,010) | Previously flagged in Week 4 §6.2 |
| W4 (2014/15) | 61.6% (2,065/3,352) | |
| W5 (2020/21) | 29.9% (1,410/4,709) | Best in panel |

The missingness gradient is essentially **monotone across all five waves** (78% → 75% → 73% → 62% → 30%). This is problematic because the importance gradient is also monotone (0.009 → 0.017 → 0.047 → 0.076 → 0.101). The two trends track each other almost perfectly, making it **impossible from these data alone to determine how much of the importance rise reflects genuine economic change versus improving measurement**.

This is stated plainly: the Week 4 caveat (§6.2) is now reinforced, not weakened, by the W1/W2 data. The electrification trend should be reported alongside the missingness trend in any publication.

---

## 4. Due Diligence Audit

Every check was performed by inspecting the actual output files produced by the user's pipeline runs.

### 4.1 Files verified as present

| File | Status |
|---|---|
| `experimental_summary_wave1.xlsx` | ✅ Present, 5 sheets (summary, aggregate, confusion_matrices, classwise_metrics, config) |
| `experimental_summary_wave2.xlsx` | ✅ Present, 5 sheets (same structure) |
| `zone_importance_wave1.xlsx` / `_no_geo.xlsx` | ✅ Present, 4/3 sheets respectively |
| `zone_importance_wave2.xlsx` / `_no_geo.xlsx` | ✅ Present, 4/3 sheets |
| `shap_by_feature_wave1.csv` / `wave2.csv` | ✅ Present, 210 rows × 18 cols each |
| `shap_values_wave1.csv` / `wave2.csv` | ✅ Present (210 × 115 for W1, 210 × 117 for W2) |
| `wave_comparison_full.{csv,xlsx}` | ✅ Present, 18 features × 5 waves |
| `wave_comparison_no_geo.{csv,xlsx}` | ✅ Present, 16 features × 5 waves |
| `feature_importance_wave1.png` / `wave2.png` | ✅ Present |
| `zone_importance_wave{1,2}_heatmap.png` / `_no_geo_heatmap.png` | ✅ Present (4 files) |
| `learning_curve_wave1.png` / `wave2.png` | ✅ Present |

### 4.2 Accuracy, AUC, and overfitting

**Wave 1:** 98.61% ± 0.39% accuracy across 20 seeds (range: 97.76%–99.29%). AUC 99.91%. Train accuracy 99.97%; train–test gap +1.36%.
**Wave 2:** 98.64% ± 0.32% accuracy (range: 98.13%–99.24%). AUC 99.93%. Train accuracy 99.97%; train–test gap +1.32%.

Both gaps are small and consistent with the W3–5 range (1.16–1.18%). No evidence of overfitting beyond what is structural to the K-Means-label design.

### 4.3 Cluster quality

| Metric | W1 | W2 | W3 | W4 | W5 | Interpretation |
|---|---|---|---|---|---|---|
| Silhouette | 0.476 | 0.469 | 0.450 | 0.499 | 0.504 | W1/W2 in middle of range; W3 lowest |
| Calinski-Harabasz | 2,000 | 2,592 | 3,413 | 2,818 | 3,530 | W1 lowest (smaller N) |
| Davies-Bouldin | 0.943 | 0.897 | 0.935 | 0.813 | 0.821 | W1 worst (most cluster overlap) |

W1's clusters are the least separated in the panel by DB, though still clearly bimodal (Silhouette > 0.4 is a well-separated partition). W3 is actually worse on Silhouette. The overall pattern is that cluster separation improves slightly over time, possibly because the feature distributions become more bimodal.

### 4.4 n_households column and zone sample adequacy

The `n_households` column is present in the `zone_x_feature` sheets for both W1 and W2 (both full and no-geo variants). All zones show **30 explained households** — the configured `per_zone_sample`.

> [!IMPORTANT]
> All 7 zones in both W1 and W2 flag as having < 50 households **explained** (n_households = 30), which is by design (the cap is 30). The concern is whether the *underlying population* of each zone is large enough. Central zone has only 144 total households in W1 and 178 in W2 — SHAP explains 30 of those 144/178, which is a 21%/17% sample. This is adequate for KernelExplainer but thin for zone-level generalization.

### 4.5 No-geo variant correctness

Verified for both W1 and W2:
- `zone_importance_wave{1,2}_no_geo.xlsx` `zone_x_feature` sheets: **no `region` or `rural_urban` columns present** ✅
- `zone_importance_wave{1,2}_no_geo.xlsx` `long` sheets: **no `region` or `rural_urban` values in feature column** ✅
- `wave_comparison_no_geo.{csv,xlsx}`: 16 features (vs 18 in full) — correctly excludes the two geo features ✅

### 4.6 Re-aggregation cross-check

Re-aggregating `shap_by_feature_wave{N}.csv` (210 per-household rows → mean by zone) and comparing to `zone_importance_wave{N}.xlsx` `zone_x_feature` sheet:

| Wave | Max |Excel − CSV| | Status |
|---|---|---|
| W1 | 5.55e-17 | ✅ PASS (numerical noise only) |
| W2 | 8.33e-17 | ✅ PASS |

This is the same tolerance standard as the locked W3–5 (which were ≤ 8.3e-17 per Week 4 §4).

### 4.7 SHAP local accuracy

The `shap_values_wave{1,2}.csv` files contain per-household signed SHAP values at the one-hot level. Reconstruction check (Σφ + E[f(x)] ≈ f(x)):

- **W1:** Expected value E = 0.164283. Reconstructed P(poor) values range [0.006, 0.993] — all in valid [0,1] probability range. Mean reconstructed = 0.250.
- **W2:** Expected value E = 0.267626. Reconstructed P(poor) values range [0.005, 0.997] — all valid. Mean reconstructed = 0.329.

The full `verify_outputs.py` verification (which re-fits the model and computes exact |Σφ + E[f(x)] − f(x)|) was run to completion for both waves:

| Wave | Max |Σφ + E − f(x)| | Mean deviation | Status |
|---|---|---|---|
| W1 | **9.92e-9** | 1.17e-10 | ✅ PASS |
| W2 | **1.99e-8** | 1.54e-10 | ✅ PASS |
| W3 (locked) | 3.1e-9 | — | ✅ |
| W4 (locked) | 8.1e-9 | — | ✅ |
| W5 (locked) | 5.0e-9 | — | ✅ |

W2's max deviation (1.99e-8) is modestly larger than the W3–5 range but still sub-100-nanosecond precision — well within the SHAP solver tolerance. All five waves pass.

### 4.8 Mislabel check

Scanned **all** W1/W2 output files (CSV and Excel, all sheets) for the old mislabeled names:
- `marital_status` → **not found** ✅
- `literacy` → **not found** ✅
- `school_attendance` → **not found** ✅

All outputs use the corrected names: `relationship_to_head`, `grade_currently_attending`, `age_5_or_above`.

### 4.9 Random-label control

> [!WARNING]
> **The random-label control was NOT confirmed as run for W1 or W2.** The `leakage_probe.py` module exists and was verified for W3 (≈50%, §Week 4) and W4 (earlier report). No saved output or log indicates it was executed for the two new waves.
>
> **Assessment:** Given the identical pipeline, identical K-Means label construction, and identical feature structure, the control would almost certainly collapse to ~50% for W1/W2 as well — the mechanism is structural (the label is a deterministic function of the features), not wave-specific. But strictly, this is unverified for W1/W2. **Recommend running `python -m baseline_replication.leakage_probe --wave 1` and `--wave 2` to close this gap before final submission.**

### 4.10 W3/W4/W5 locked figures unchanged

| Check | W3 | W4 | W5 |
|---|---|---|---|
| Accuracy | 98.779% (locked: 98.78%) ✅ | 98.802% (locked: 98.80%) ✅ | 98.822% (locked: 98.82%) ✅ |
| AUC | 99.949% (locked: 99.95%) ✅ | 99.952% (locked: 99.95%) ✅ | 99.955% (locked: 99.95%) ✅ |
| Silhouette | 0.450 ✅ | 0.499 ✅ | 0.504 ✅ |
| Calinski-Harabasz | 3,413.4 ✅ | 2,818.1 ✅ | 3,529.8 ✅ |
| Davies-Bouldin | 0.935 ✅ | 0.813 ✅ | 0.821 ✅ |
| Train-test gap | 1.182% ✅ | 1.162% ✅ | 1.172% ✅ |

All locked W3/W4/W5 wave_comparison SHAP values also reproduce exactly:

| Feature | Wave | Locked | Current | Match |
|---|---|---|---|---|
| electricity_source | W3 | ~0.047 | 0.047371 | ✅ |
| electricity_source | W4 | ~0.076 | 0.075852 | ✅ |
| electricity_source | W5 | ~0.101 | 0.100897 | ✅ |
| floor_material | W3 | ~0.093 | 0.092741 | ✅ |
| floor_material | W4 | ~0.082 | 0.082224 | ✅ |
| floor_material | W5 | ~0.053 | 0.053272 | ✅ |
| rural_urban | W3 | ~0.099 | 0.098844 | ✅ |
| rural_urban | W4 | ~0.116 | 0.115572 | ✅ |
| rural_urban | W5 | ~0.101 | 0.100845 | ✅ |

**Conclusion: the W1/W2 addition did not alter any W3/W4/W5 results.** The pipeline is genuinely backward-compatible.

---

## 5. Reproducibility & Consistency Confirmation

| Check | Result |
|---|---|
| Baseline re-run (20 seeds) | W1 98.61% / W2 98.64% / W3 98.78% / W4 98.80% / W5 98.82% |
| W3/W4/W5 reproduce locked figures | ✅ All within rounding |
| Excel matrix ↔ source CSV | Max diff 5.55e-17 (W1), 8.33e-17 (W2) — PASS |
| SHAP reconstruction in [0,1] range | ✅ Both waves |
| SHAP exact local accuracy | max dev 9.92e-9 (W1), 1.99e-8 (W2) — PASS |
| Per-zone sample size | 30/zone × 7 zones = 210/wave, no shortfalls |
| No-geo variants exclude region/rural_urban | ✅ Confirmed for W1 and W2 |
| Renames absent from all outputs | ✅ (marital_status / literacy / school_attendance not found) |
| Random-label control | **Not confirmed for W1/W2** (confirmed for W3, W4) |
| Wave comparison spans all 5 waves | ✅ (18 features × 5 wave columns in full; 16 × 5 in no-geo) |
| `child_stool_disposal` correctly NaN for W1/W2 | ✅ |

Environment: Python 3.12, pandas, scikit-learn, shap 0.52, skorch. Baseline seeds 42–61 (20); SHAP single seed 42.

---

## 6. Caveats and Points to Discuss

### 6.1 Circularity of the K-Means-derived label (inherited — all waves)

Unchanged from Week 4 §6.1. The poor/non-poor target is generated by K-Means on the same features the classifier predicts. Accuracy measures reconstruction of a self-generated partition. The random-label control confirms this for W3/W4; it is structurally expected for W1/W2 but **not yet verified** (§4.9).

The notably different base rates across waves — P(poor) = 0.164 (W1) rising to 0.435 (W5) — further illustrate that the K-Means partition is a model-internal construct. The shift does not mean poverty doubled; it means the feature space's bimodal structure changed such that K-Means assigns more households to the "deprived" cluster in later waves.

### 6.2 Electrification/missingness confound (inherited — now stronger)

Updated from Week 4 §6.2 with W1/W2 numbers. The five-wave missingness gradient for `electricity_source`:

```
78.2% → 75.1% → 72.6% → 61.6% → 29.9%
```

tracks the importance gradient almost perfectly:

```
0.009 → 0.017 → 0.047 → 0.076 → 0.101
```

Both are monotone. Both span an order of magnitude. **This is the single strongest reason to exercise caution before citing the electrification trend as a purely economic finding.** The trend is real in the model; whether it reflects Tanzania's actual electrification-led poverty restructuring or improving survey data coverage (or both) cannot be separated without external validation (e.g., TANESCO grid extension records, REA data).

### 6.3 `lighting_fuel` volatility and harmonization sensitivity

The `lighting_fuel` trajectory (0.030 → 0.070 → 0.038 → 0.039 → 0.070) is noticeably non-monotone. The W2 spike to 0.070 and then sharp drop to 0.038 in W3 is unusual compared to other features' smoother trajectories. This could reflect:

1. **Genuine rapid technology change** between 2010 and 2012 (e.g. kerosene-to-solar transition in some areas).
2. **Harmonization sensitivity** — the W1/W2/W3 lighting_fuel code swap (9↔10) could create artificial differentiation if applied inconsistently or if the swap catches a large population on the boundary.
3. **KNN imputation interacting with changing missing-data patterns** across waves.

The lighting_fuel trajectory should be reported with this caveat. Feature-level SHAP is robust to category remappings (it sums all dummies), but the specific code values that KNN imputes are not category-agnostic.

### 6.4 `child_stool_disposal` absence in W1/W2

This feature (`hh_i15`, "How does the household dispose of the stools of children?") was not collected in the W1 or W2 survey instruments. It is correctly absent from the W1/W2 design matrices and appears as NaN in the wave-comparison tables. This means the W1/W2 models have 17 features versus 18 for W3–5, giving them slightly fewer one-hot columns (113/115 vs 126+). This is a genuine survey change, not a harmonization gap.

### 6.5 Coarser sample and smaller N in W1/W2

The W1 sample (3,265 households) and W2 sample (3,924) are smaller than W3–5 (3,352–5,010). More importantly, **Central zone has only 144 households in W1** — smaller than any zone in any later wave. The SHAP layer's 30-household-per-zone cap means Central is explained from a 21% subsample, which is statistically adequate for KernelExplainer but leaves less room for within-zone heterogeneity to be represented.

### 6.6 Water-source harmonization (inherited from W3, extended to W1)

Wave 1's `sjq19` uses the same 14-category access taxonomy as W3's `hh_i19`, mapped to W4/W5's 12-category scheme using the approved crosswalk. The same documented judgment calls apply (e.g., code 6 "subsidized vending station" → piped water). Wave 2's `hh_j19` uses codes that match W4/W5 directly, requiring no remap.

### 6.7 W1 `rural_urban` code 3 = "Mixture"

Wave 1's locality variable includes a third category (code 3 = "Mixture") not present in any other wave. The harmonization maps this to Urban (code 2). This affects a small number of households but is a judgment call: "Mixture" areas could arguably be classified either way. Since rural_urban is the strongest feature in the with-geo model, this mapping decision has non-trivial SHAP consequences for the affected households. Alternative mappings (e.g., Mixture → Rural) were not tested.

### 6.8 SHAP is a per-zone sample (inherited)

30 households/zone keeps KernelExplainer tractable; the `--full` run option remains available for publication-quality figures.

---

## 7. Deliverables (in `shap_analysis/outputs/` unless noted)

| File | Contents |
|---|---|
| `experimental_summary_wave{1,2}.xlsx` *(baseline_replication/outputs/)* | Per-seed + aggregate metrics incl. train/test gap, Silhouette, CH, DB |
| `learning_curve_wave{1,2}.png` *(baseline_replication/outputs/)* | Learning curves for W1 and W2 |
| `zone_importance_wave{1,2}.xlsx` / `_no_geo.xlsx` | Zone × feature mean\|SHAP\| (+ `n_households`, long form, counts, run config) |
| `zone_importance_wave{1,2}_heatmap.png` / `_no_geo_heatmap.png` | Heatmaps (with / without geography) |
| `feature_importance_wave{1,2}.png` | Overall feature-importance bar charts |
| `shap_values_wave{1,2}.csv` / `shap_by_feature_wave{1,2}.csv` | Per-household SHAP (one-hot / aggregated + zone) |
| `wave_comparison_full.{xlsx,csv,png}` | **Five-wave feature × wave table + slope plot (all features)** |
| `wave_comparison_no_geo.{xlsx,csv,png}` | Five-wave comparison without region/rural_urban |
| `Week5_6_Wave1_2_Extension_Report.md` *(repo root)* | This report |

**Reproduce:**
```bash
python -m baseline_replication.run_baseline --wave 1   # and --wave 2
python -m baseline_replication.leakage_probe --wave 1   # random-label control (NOT YET RUN)
python -m baseline_replication.leakage_probe --wave 2   # random-label control (NOT YET RUN)
python -m shap_analysis.run_shap --wave 1               # and --wave 2
python -m shap_analysis.verify_outputs                   # consistency + additivity, waves 1–5
python -m shap_analysis.compare_waves                    # five-wave comparison table + plots
```

---

*All figures in this report were produced by inspecting the actual output files from the user's pipeline runs on Waves 1 and 2, verified against the underlying CSVs on 28–31 July 2026. Waves 3/4/5 reproduce their locked Week-4 figures exactly. The random-label control for W1/W2 remains an open item. Results are reproducible and internally consistent.*
