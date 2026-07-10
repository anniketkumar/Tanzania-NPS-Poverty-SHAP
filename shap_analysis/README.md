# Zone-stratified SHAP (Tanzania NPS) — Week 3

The Week-3 deliverable of *"Explaining Poverty Clusters Across Space and Time"*:
attach **SHAP** to the Week 1–2 baseline and compute **zone-stratified feature
importance** for Waves 4 and 5.

> Proposal §4: *"Attach SHAP (KernelExplainer on no-PCA model). Compute
> zone-stratified feature importance for both waves. Deliverable: SHAP values
> matrix + zone-importance heatmap."*

```
                          baseline_replication (locked)
raw NPS CSVs ─▶ features ─▶ one-hot ─▶ KNN impute ─▶ PCA ─▶ K-Means ─▶ Stacked Ensemble
                                          │                    │            │
                                          │            (PCA only makes       │
                                          │             the K=2 label)       │
                                          └──────────── raw features ────────┘
                                                                             │
                                                    shap_analysis ▶ KernelExplainer on P(poor)
                                                        ▶ aggregate one-hot → 18 features
                                                        ▶ group by zone → zone × feature heatmap
```

This package **imports** the baseline and never modifies it — the baseline is the
locked Week 1–2 replication, and its README notes SHAP *"slots in after this
baseline is locked … on the `classifier_input="raw"` variant."* That raw-feature
model is exactly the **no-PCA model** the task asks for: SHAP values map to real
survey questions (floor material, toilet type, …), not principal components. PCA
is still used, but only to manufacture the K-Means poor/non-poor label — the
classifier and SHAP never see the components.

---

## Quick start

```bash
# from the project root (folder containing baseline_replication/ and shap_analysis/)
pip install -r shap_analysis/requirements.txt

python -m shap_analysis.run_shap --wave 4      # sampled (fast, minutes)
python -m shap_analysis.run_shap --wave 5
```

Useful flags:

| Flag | Effect |
|---|---|
| `--full` | Explain **every** household (the complete SHAP matrix; slow — for the eventual paper run). |
| `--per-zone N` | Households explained per zone in sampled mode (default 40). |
| `--background N` | k-means background summary size for SHAP (default 50). |
| `--no-zanzibar` | Restrict to the 6 mainland zones (drop Zanzibar households). |
| `--seed S` | Seed for the single fitted model (default 42). |
| `--no-plots` | Skip the PNGs. |

### Why sampled by default

`shap.KernelExplainer` is **model-agnostic** — the only option here, since the
stack contains a skorch MLP and a logistic meta-learner (so the fast TreeExplainer
does not apply). It is also slow: cost grows with (# households × background size
× perturbations). For a zone **mean** we don't need every household, so the
default summarises the background with k-means (~50 rows) and explains a
stratified per-zone sample (~40/zone) — minutes, not hours. Use `--full` for the
complete per-household matrix.

---

## Outputs (`shap_analysis/outputs/`)

| File | What it is |
|---|---|
| `zone_importance_wave{N}.xlsx` | **The SHAP matrix deliverable** — zone × feature mean\|SHAP\| (`zone_x_feature` sheet), a tidy `long` sheet, per-zone household counts (`zone_sample_sizes`), and the run config. |
| `zone_importance_wave{N}_heatmap.png` | **The heatmap deliverable** — zones × features, colour = mean\|SHAP\|. |
| `zone_importance_wave{N}_no_geo.xlsx` | Zone importance **without** `region` and `rural_urban` (tautology-check variant). |
| `zone_importance_wave{N}_no_geo_heatmap.png` | Heatmap **without** geographic features — for discussing the circularity risk. |
| `feature_importance_wave{N}.png` | Overall (zone-averaged) feature-importance bar chart. |
| `shap_values_wave{N}.csv` | Per-household signed SHAP over the raw one-hot columns. |
| `shap_by_feature_wave{N}.csv` | Per-household \|SHAP\| aggregated to the 18 features, with each household's zone. |

Reading the heatmap answers the proposal's policy question directly: *"in the
Western zone, should limited budget go to electrification, schools, or water?"*

---

## Module layout

```
shap_analysis/
├── config.py     # SHAPConfig: wave, seed, background/sample sizes, --full, zanzibar toggle.
│                 #   Inherits the baseline Config so the pipeline stays faithful to Week 1-2.
├── zones.py      # REGION_TO_ZONE — the paper's 7 zones (single source of truth). See below.
├── model.py      # fit_full_model(): baseline pipeline on the FULL wave, single seed.
├── explain.py    # run_shap(): KernelExplainer → one-hot→feature aggregation → zone matrix.
├── plots.py      # zone-importance heatmap + overall bar chart.
└── run_shap.py   # CLI: fit → explain → save.
```

---

## The region → zone mapping (read this)

The paper (Sende et al. 2025, §V.B) reports poverty over **7 zones**: Western,
Lake, Central, Southern, Northern, Coastal, **Zanzibar** (Zanzibar is one of the
7, not an 8th). I match that taxonomy so the SHAP output lines up with the paper's
zone-level poverty rates.

**Assumption to verify:** the paper does not publish an explicit region → zone
table. `zones.py` uses a documented, standard NBS grouping. Two assignments are
genuinely ambiguous and are the ones to confirm with the authors:

1. **Southern Highlands** (Iringa, Mbeya, Rukwa, Njombe, Songwe) is folded into
   `Southern` (the paper lists no separate "Southern Highlands" among its 7).
2. **Shinyanga** is placed in `Lake` (its modern neighbourhood); some older
   schemes put it in `Western`.

Both are one-line edits in `zones.py::REGION_TO_ZONE`.

## Notes

- **Single-seed model.** SHAP explains one fitted model, so this uses a single
  seed on the full data rather than the baseline's 20-seed evaluation loop.
- **Class explained.** SHAP attributes the probability of the **"poor"** cluster
  (label 1, the more-deprived centroid under the baseline's label ordering);
  change via `SHAPConfig.explain_class`.
- **One-hot → feature aggregation** sums \|SHAP\| across all dummies of a survey
  question (approved in `Baseline_Replication_Report.md` §6).
