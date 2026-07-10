"""Week 3 — zone-stratified SHAP analysis of the poverty-clustering pipeline.

This package attaches a SHAP explainability layer on top of the Week 1-2 baseline
(`baseline_replication/`). It answers the policy-facing question the black-box
accuracy cannot: *which deprivation dimensions drive the poverty classification
in each of Tanzania's zones?*

Pipeline reused from the baseline (no PCA on the classifier path):

    raw NPS CSVs -> ~18 features -> one-hot -> KNN impute ->  ... -> Stacked Ensemble
                                                    |                       |
                                                    +---- SHAP explains ----+
                                              (raw imputed feature space, not PCA)

See README.md for the run commands and the note on the region->zone mapping.
"""

__version__ = "0.1.0"
