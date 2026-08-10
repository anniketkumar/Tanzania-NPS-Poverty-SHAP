# Week 4 — Wave 3 Harmonization Crosswalk (for sign-off before recoding)

**Project:** *Spatial Distribution of Poverty Clusters and Its Prediction Algorithms* — Wave 3 (2012/13) extension
**Replication base:** Sende et al. (2025, IEEE Access)
**Prepared by:** Aniket Kumar Pradhan, NTU Singapore
**Supervisor:** Prof. Snehanshu Saha, APPCAIR, BITS Pilani Goa
**Date:** 14 July 2026
**Status:** ⏸ **AWAITING APPROVAL** — nothing has been recoded yet. This table is presented per the Week 4 Step 1 requirement ("show it to me before applying it — don't silently recode").

---

## 0. Method

Value labels were pulled directly from `converted data/master_data_dictionary.csv` (`value_labels`
JSON column), which is derived from the original Stata `.dta` / SPSS `.sav` binaries, for **every**
feature in the baseline `feature_spec.py` — not only those flagged `verify=True`. Each Wave 3 code
set was compared cell-by-cell against Waves 4 and 5. Harmonization target = **the Wave 4/5 reference
scheme** (so the locked W4/W5 encodings are left untouched; only Wave 3 is recoded).

**Bottom line:** only **two** Wave 3 features actually need recoding — `water_source` (major) and
`lighting_fuel` (a one-line code swap the spec never flagged). Everything else in Wave 3 matches
Wave 4 code-for-code. Most cross-wave divergence lives on the **Wave 5** side (already locked).

---

## 1. `water_source` (`hh_i19`) — MAJOR recode

Wave 3 uses a 14-category *access/delivery* taxonomy; Waves 4/5 use a 12-category *source-type*
taxonomy. They are structurally different, so this is a judgement-based crosswalk, not a relabel.
W3 missingness: 1 / 5010.

| W3 code | W3 label | → W4/W5 code | W4/W5 label | Note |
|--------:|----------|-------------:|-------------|------|
| 1 | PIPED WATER INSIDE DWELLING | 1 | PIPED WATER | direct |
| 2 | PRIVATE OUTSIDE STANDPIPE/TAP | 1 | PIPED WATER | piped to yard/plot = piped (JMP) |
| 3 | PUBLIC STANDPIPE/TAP | 1 | PIPED WATER | public tap = improved piped (JMP) |
| 4 | NEIGHBOURING HOUSEHOLD | 1 | PIPED WATER | ⚠️ **judgement** — "piped from neighbour" (JMP); alt = Other |
| 5 | WATER VENDOR | 9 | CART WITH SMALL TANK/DRUM | vendor delivery ≈ cart/small tank |
| 6 | SUBSIDIZED WATER VENDING STATION | 1 | PIPED WATER | fixed kiosk fed by piped network, sold by the bucket — structurally a public standpipe (code 3), not mobile delivery *(supervisor decision, 14 Jul 2026)* |
| 7 | WATER TRUCK/TANKER SERVICE | 10 | TANKER-TRUCK | direct |
| 8 | PROTECTED WELL WITH PUMP | 2 | TUBEWELL / BOREHOLE | ⚠️ **judgement** — "with pump" ⇒ mechanized/borehole; alt = 3 PROTECTED DUG WELL |
| 9 | UNPROTECTED WELL WITH PUMP | 4 | UNPROTECTED DUGWELL | ⚠️ **judgement** — unprotected trumps pump; alt = 2 |
| 10 | PROTECTED WELL WITHOUT PUMP | 3 | PROTECTED DUG WELL | direct concept |
| 11 | UNPROTECTED WELL WITHOUT PUMP | 4 | UNPROTECTED DUGWELL | direct concept |
| 12 | RIVER, LAKE, SPRING, POND | 11 | SURFACE WATER | springs folded into surface (W3 cannot separate protected/unprotected spring) |
| 13 | RAINWATER | 7 | RAINWATER COLLECTION | direct |
| 14 | OTHER (SPECIFY) | 12 | OTHER, SPECIFY | direct |

W4/W5 categories **5 PROTECTED SPRING, 6 UNPROTECTED SPRING, 8 BOTTLED WATER** have no Wave 3
equivalent and will simply be empty in Wave 3.

**Four judgement calls to confirm: codes 4, 6, 8, 9.** If you prefer, an alternative is a coarser
common taxonomy (Piped/Tap · Borehole/Protected-well · Unprotected-well · Surface · Rainwater ·
Delivered · Other) applied to all waves — cleaner semantics but it would recode the locked W4/W5,
so it is **not** my recommendation. Feature-level SHAP is category-agnostic either way.

## 2. `lighting_fuel` (`hh_i17`) — MINOR recode (⚠️ not flagged in the spec)

Codes 9 and 10 are **swapped** between Wave 3 and Waves 4/5. This feature is **not** marked
`verify=True` — it would have been recoded silently without this check.

| Code | Wave 3 | Waves 4/5 | Action |
|-----:|--------|-----------|--------|
| 9 | OTHER (SPECIFY) | TORCH | swap |
| 10 | TORCH/TOCHI | OTHER (SPECIFY) | swap |

Codes 1–8 identical across all waves. **Proposed W3 recode:** `9 → 10`, `10 → 9` (map W3 to the
W4/W5 code positions). W3 missingness: 1 / 5010.

## 3. `rural_urban` — normalization (loader fallback)

`rural_urban` was silently dropped in W3 & W5 (loader points only at `clustertype`, which is W4-only).
Recovering it requires a `y{wave}_rural` fallback and code normalization, since the three sources
disagree:

| Wave | Source column | Raw codes | → canonical |
|------|---------------|-----------|-------------|
| W3 | `y3_rural` (0 miss / 5010) | 0=URBAN, 1=RURAL | 1=RURAL, 2=URBAN |
| W4 | `clustertype` (0 miss; `y4_rural` is 100% missing) | 1=RURAL, 2=URBAN | 1=RURAL, 2=URBAN |
| W5 | `y5_rural` (0 miss / 4709) | 1=RURAL, 2=URBAN | 1=RURAL, 2=URBAN |

W3 distribution: RURAL 3219 / URBAN 1791.

---

## 4. Features confirmed IDENTICAL in Wave 3 vs Wave 4 — **no recode**

`head_sex` (hh_b02), `head_age` (hh_b04, numeric), `relationship` (hh_b05), `school_attendance`
(hh_c01), `education_level` (hh_c07), `literacy` (hh_c09), `housing_tenure` (hh_i01), `rooms`
(hh_i09, numeric), `floor_material` (hh_i10), `garbage_disposal` (hh_i11), `toilet_facility`
(hh_i12), `electricity_source` (hh_i18). All share W4's exact code sets.

## 5. Flags to carry into the report (do NOT affect the Wave 3 recode)

These are silent shifts / mislabels found during the sweep. Most sit on the **locked Wave 5** side
or are naming issues; none require touching Wave 3 beyond §1–§3, but they belong in the caveats.

- **Spec mislabels (all waves, codes consistent — usable, names wrong):**
  `marital_status` → `hh_b05` is *relationship to head*; `literacy` → `hh_c09` is *grade currently
  attending*; `school_attendance` → `hh_c01` is *"is [NAME] 5 years or above?"*.
- **`garbage_disposal` expanded 6 → 11 categories in Wave 5** (W3 = W4 = 6). Not flagged `verify`.
  Affects locked W5 only.
- **`floor_material` expanded 3 → 6 in Wave 5** (W3 = W4 = 3). Already flagged `verify`; W3 needs no recode.
- **`education_level` value codes shifted in Wave 5** (e.g. W5 `2=MS+COURSE`, `11=ADULT`; W3 = W4).
  Already flagged; W3 needs no recode.
- **`toilet_facility` code 2 reworded in W5** ("PIT LATRINE WITHOUT SLAB/OPEN PIT" vs W3/W4 "OPEN PIT
  WITHOUT SLAB") — same concept, no count change.
- **`child_stool_disposal`:** W3 has 8 categories vs 10 in W4/W5 (W3 lacks the two diaper options) —
  a clean subset, no recode.
- **`electricity_source` missingness is worst in Wave 3: 72.6% (3635/5010)** vs W4 61.6% vs W5 30%.
  Relies entirely on KNN imputation — a Wave 3 data-quality caveat.
- **Zone check passed:** all W3 region codes (1–21, 51–55) map via `zones.py`; splits 22–26 absent in W3.

---

## Requested sign-off

Please confirm (or amend) **§1 water_source** (esp. the four ⚠️ judgement calls: codes 4, 6, 8, 9),
**§2 lighting_fuel** swap, and the **§3 rural_urban** canonicalization. On approval I will implement
these in a new `harmonize.py` (W3-only maps; W4/W5 identity) and proceed to Step 2.
