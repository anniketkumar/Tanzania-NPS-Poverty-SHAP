"""Baseline replication of the Sende et al. (2025) spatial-poverty pipeline.

Pipeline:  KNN imputation -> PCA -> K-Means -> Stacked Ensemble
Stack:     scikit-learn StackingClassifier (PyTorch/skorch MLP as one base learner)

See README.md for how to run and for the note on what the accuracy really means.
"""

__version__ = "0.1.0"
