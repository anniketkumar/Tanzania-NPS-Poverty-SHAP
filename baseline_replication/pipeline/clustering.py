"""K-Means clustering — this is the step that actually manufactures the label.

I fit K-Means on the reduced training data. If I turn on auto-K it tries a few
values and keeps whichever gives the best silhouette score. Because K-Means
hands out cluster numbers arbitrarily (which group gets called "0" flips from
seed to seed), I re-number them by how far each centroid sits from the origin.
That way "0" and "1" refer to the same kind of cluster on every run, so I can
line up per-class metrics across seeds without them getting scrambled.

One thing to keep in mind: the label this produces is purely a function of the
features (via scale -> PCA -> nearest centroid). Nothing external goes into it.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..config import ClusterConfig


def select_k_by_silhouette(
    X: np.ndarray, k_range: tuple[int, ...], seed: int, n_init: int
) -> tuple[int, dict[int, float]]:
    """Return the K in `k_range` with the highest silhouette score."""
    scores: dict[int, float] = {}
    for k in k_range:
        if k < 2 or k >= len(X):
            continue
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    return best_k, scores


class PovertyClusterer:
    """A fitted KMeans plus the lookup that re-numbers its raw cluster ids."""

    def __init__(self, kmeans: KMeans, label_map: dict[int, int]):
        self.kmeans = kmeans
        self.label_map = label_map

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.kmeans.predict(X)
        return np.vectorize(self.label_map.get)(raw)


def fit_clusterer(
    X_train_reduced: np.ndarray, cfg: ClusterConfig, seed: int
) -> tuple[PovertyClusterer, dict]:
    """Fit K-Means on the reduced training data and build a stable label map."""
    info: dict = {}

    if cfg.auto_select_k:
        k, sil_scores = select_k_by_silhouette(
            X_train_reduced, cfg.k_search_range, seed, cfg.n_init
        )
        info["silhouette_scores"] = sil_scores
    else:
        k = cfg.n_clusters
    info["k"] = k

    km = KMeans(n_clusters=k, n_init=cfg.n_init, random_state=seed)
    raw_labels = km.fit_predict(X_train_reduced)

    if cfg.order_labels_by_deprivation:
        # Sort the clusters by how far their centroid is from the origin (its L2
        # norm) in PCA space, then renumber them in that order. This just gives a
        # stable, seed-independent naming; it's a proxy for "how deprived", so if
        # domain knowledge says the ordering should flip, flip it here.
        order = np.argsort(np.linalg.norm(km.cluster_centers_, axis=1))
        label_map = {int(old): int(new) for new, old in enumerate(order)}
    else:
        label_map = {i: i for i in range(k)}
    info["label_map"] = label_map

    clusterer = PovertyClusterer(km, label_map)
    train_labels = clusterer.predict(X_train_reduced)
    info["train_label_counts"] = dict(
        zip(*np.unique(train_labels, return_counts=True))
    )
    return clusterer, info
