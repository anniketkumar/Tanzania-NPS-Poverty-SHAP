"""A quick end-to-end test on made-up data — no real survey files needed.

I build a small fake dataset with two obvious groups and a few missing cells,
then run the whole KNN -> PCA -> K-Means -> Stacked Ensemble pipeline for a
couple of seeds with a tiny, fast network. It's just a wiring check: does the
whole thing run and produce sensible numbers? I run this before pointing the
pipeline at the real wave-4 data so I catch plumbing bugs early.

    python -m aniket_code.smoke_test
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import default_config
from .pipeline import run_experiment


def make_synthetic_design(n=400, n_features=20, missing_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    # Make two well-separated blobs so K-Means has an easy, obvious split to find.
    half = n // 2
    g0 = rng.normal(-1.0, 1.0, size=(half, n_features))
    g1 = rng.normal(1.0, 1.0, size=(n - half, n_features))
    X = np.vstack([g0, g1])
    rng.shuffle(X)
    # Poke some holes in the data so the KNN imputer has something to fill.
    mask = rng.random(X.shape) < missing_frac
    X[mask] = np.nan
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols)


def main() -> int:
    design = make_synthetic_design()

    cfg = default_config()
    cfg.experiment.seeds = (42, 43)        # just two seeds
    cfg.experiment.make_plots = False
    cfg.nn.max_epochs = 5                  # keep the NN fast
    cfg.nn.hidden_dims = (8,)
    cfg.cluster.n_clusters = 2

    results = run_experiment(design, y_ext=None, cfg=cfg)

    summary = results["summary"]
    print("\n--- per-seed summary ---")
    print(summary.to_string(index=False))
    print("\n--- aggregate ---")
    print(results["aggregate"].to_string(index=False))

    # Sanity check, not a performance claim. The label here is just the K-Means
    # cluster, which is a deterministic function of the features, so the
    # classifier can basically reconstruct it and accuracy comes out very high.
    # That's expected and is exactly the circularity to keep in mind when reading
    # the accuracy on the real data too.
    assert summary["accuracy"].mean() > 0.8, "Unexpectedly low accuracy"
    assert summary["auc_macro"].notna().all(), "AUC failed to compute"
    print("\nSMOKE TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
