"""Step-4 sanity check: internal consistency + SHAP additivity.

Run AFTER `run_shap.py` has produced the wave-3/4/5 artifacts. It re-derives the
model deterministically (seed 42, same as the deliverable run) and checks:

  (a) every zone_importance_*.xlsx carries an `n_households` column and no zone
      fell below the requested per-zone cap;
  (b) the no-geo files/heatmaps really drop region + rural_urban;
  (c) the zone_x_feature matrix in each Excel matches what shap_by_feature CSV
      re-aggregates to (Excel <-> CSV consistency);
  (d) SHAP local accuracy: sum(signed SHAP over one-hot) + expected_value
      ~= model P(poor) for a random sample of households, per wave.

Prints a compact PASS/FAIL report and the max additivity deviation per wave.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import default_config
from .explain import _stratified_foreground, run_shap  # noqa: F401  (run_shap for parity)
from .model import fit_full_model
from .zones import region_code_to_zone

GEO = ["region", "rural_urban"]
OD = default_config().output_dir


def _load_expected_value(wave: int) -> float:
    rc = pd.read_excel(OD / f"zone_importance_wave{wave}.xlsx", sheet_name="run_config")
    return float(rc.set_index("setting").loc["expected_value", "value"])


def check_wave(wave: int) -> dict:
    out = {"wave": wave, "problems": []}

    def prob(msg):
        out["problems"].append(msg)

    # ---- Excel structure: n_households + geo columns --------------------- #
    main = pd.read_excel(OD / f"zone_importance_wave{wave}.xlsx",
                         sheet_name="zone_x_feature", index_col=0)
    nogeo = pd.read_excel(OD / f"zone_importance_wave{wave}_no_geo.xlsx",
                          sheet_name="zone_x_feature", index_col=0)

    cfg = default_config()
    cfg.wave = wave
    cap = cfg.per_zone_sample

    for name, df in [("main", main), ("no_geo", nogeo)]:
        if "n_households" not in df.columns:
            prob(f"{name}: missing n_households column")
        else:
            nvals = df["n_households"]
            out[f"{name}_n_per_zone"] = nvals.to_dict()
            if (nvals < cap).any():
                prob(f"{name}: zone(s) below cap {cap}: "
                     f"{nvals[nvals < cap].to_dict()}")
            if (nvals != nvals.iloc[0]).any():
                prob(f"{name}: uneven zone counts {nvals.to_dict()}")

    if any(g in main.columns for g in GEO):
        present = [g for g in GEO if g in main.columns]
        out["main_geo_present"] = present  # expected (main keeps geo)
    if any(g in nogeo.columns for g in GEO):
        prob(f"no_geo Excel still contains geo columns: "
             f"{[g for g in GEO if g in nogeo.columns]}")

    # ---- Excel <-> CSV consistency -------------------------------------- #
    csv = pd.read_csv(OD / f"shap_by_feature_wave{wave}.csv")
    feat_cols = [c for c in csv.columns if c != "zone"]
    re_agg = csv.groupby("zone")[feat_cols].mean()
    main_feats = [c for c in main.columns if c != "n_households"]
    common = [c for c in main_feats if c in re_agg.columns]
    aligned_main = main.loc[re_agg.index.intersection(main.index), common]
    aligned_csv = re_agg.loc[aligned_main.index, common]
    max_excel_csv = float(np.abs(aligned_main.values - aligned_csv.values).max())
    out["max_excel_vs_csv_diff"] = max_excel_csv
    if max_excel_csv > 1e-9:
        prob(f"zone_x_feature Excel disagrees with CSV re-aggregation "
             f"(max diff {max_excel_csv:.2e})")

    # ---- SHAP additivity (local accuracy) ------------------------------- #
    model = fit_full_model(cfg)
    shap_onehot = pd.read_csv(OD / f"shap_values_wave{wave}.csv", index_col=0)
    expected = _load_expected_value(wave)

    # Model P(poor) on the exact households we explained.
    rows = shap_onehot.index.to_numpy()
    X_fg = model.X_imp.loc[rows]
    p_poor = model.clf.predict_proba(X_fg.to_numpy())[:, cfg.explain_class]
    recon = shap_onehot.sum(axis=1).to_numpy() + expected
    dev = np.abs(recon - p_poor)

    rng = np.random.RandomState(0)
    sample = rng.choice(len(rows), size=min(25, len(rows)), replace=False)
    out["additivity_max_dev"] = float(dev.max())
    out["additivity_mean_dev"] = float(dev.mean())
    out["additivity_max_dev_sample"] = float(dev[sample].max())
    out["expected_value"] = expected
    out["n_explained"] = len(rows)
    # Local accuracy holds to solver tolerance; flag only gross violations.
    if dev.max() > 1e-2:
        prob(f"SHAP additivity off: max |sum(SHAP)+E - f(x)| = {dev.max():.4e}")

    return out


def main() -> int:
    all_ok = True
    for wave in (3, 4, 5):
        r = check_wave(wave)
        print(f"\n===== Wave {wave} =====")
        print(f"  n_households per zone (main):  {r.get('main_n_per_zone')}")
        print(f"  n_households per zone (no_geo): {r.get('no_geo_n_per_zone')}")
        print(f"  main geo columns present:      {r.get('main_geo_present', [])}")
        print(f"  max |Excel - CSV re-agg|:      {r['max_excel_vs_csv_diff']:.2e}")
        print(f"  SHAP expected_value:           {r['expected_value']:.6f}")
        print(f"  additivity max dev (all {r['n_explained']}):   "
              f"{r['additivity_max_dev']:.3e}")
        print(f"  additivity mean dev:           {r['additivity_mean_dev']:.3e}")
        print(f"  additivity max dev (25 rnd):   {r['additivity_max_dev_sample']:.3e}")
        if r["problems"]:
            all_ok = False
            print("  PROBLEMS:")
            for p in r["problems"]:
                print(f"    - {p}")
        else:
            print("  PASS: no problems found.")
    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
