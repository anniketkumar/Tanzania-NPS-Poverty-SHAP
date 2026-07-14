"""Build one household-level table by joining the raw survey CSVs together.

The survey splits data across sections (A, B, C, I) in separate files. This
module joins them on the household id, keeps just the head-of-household row from
the person-level sections, counts household members to get household_size, and
for wave 4 recovers rural/urban from the `clustertype` column (its dedicated
column is empty there). What comes out is one row per household, with columns
renamed to my feature names but still holding the raw survey codes — the
one-hot encoding happens later in encode.py.

I kept this explicit and free of side effects because it's the part I'll most
likely tweak when adapting to a different set of features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import feature_spec as fs
from .harmonize import harmonize_wave


def _household_dir(converted_data_dir: Path, wave: int) -> Path:
    d = Path(converted_data_dir) / f"wave{wave}" / "household"
    if not d.is_dir():
        raise FileNotFoundError(
            f"Expected household CSVs at {d}. Check config.data.converted_data_dir "
            f"and that wave {wave} has been converted."
        )
    return d


def _read_section(hh_dir: Path, section: str) -> pd.DataFrame:
    path = hh_dir / f"HH_SEC_{section}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing section file: {path}")
    # utf-8-sig strips the byte-order mark some of these converted CSVs start
    # with; low_memory=False stops pandas complaining about mixed-type columns.
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def load_wave_features(
    converted_data_dir: Path,
    wave: int,
    head_of_household_only: bool = True,
) -> pd.DataFrame:
    """Return a one-row-per-household DataFrame of raw (un-encoded) features."""
    hh_dir = _household_dir(converted_data_dir, wave)
    hhid = f"y{wave}_hhid"
    indid = f"indidy{wave}"

    sections = fs.features_by_section()

    # ---- Section A (household level) ------------------------------------- #
    sec_a = _read_section(hh_dir, "A")
    a_cols = [hhid] + [f.column for f in sections.get("A", []) if f.column in sec_a.columns]

    # rural/urban fallback: the feature spec points at `clustertype` (the W4
    # source), but that column only exists in W4. For W3/W5 fall back to the
    # wave's own y{wave}_rural column. Whichever we pick gets renamed to
    # `rural_urban` below; harmonize_wave() then normalizes the codes.
    rural_src = None
    if "clustertype" in sec_a.columns and sec_a["clustertype"].notna().any():
        rural_src = "clustertype"
    elif f"y{wave}_rural" in sec_a.columns and sec_a[f"y{wave}_rural"].notna().any():
        rural_src = f"y{wave}_rural"
    if rural_src and rural_src not in a_cols:
        a_cols.append(rural_src)

    base = sec_a[list(dict.fromkeys(a_cols))].drop_duplicates(subset=hhid).copy()

    # ---- Individual-level sections (B, C): keep the head of household ----- #
    for sec_name in ("B", "C"):
        specs = sections.get(sec_name, [])
        if not specs:
            continue
        df = _read_section(hh_dir, sec_name)
        wanted = [c for c in (f.column for f in specs) if c in df.columns]

        if sec_name == "B":
            # Count members per household BEFORE I filter down to the head,
            # otherwise every household would come out as size 1.
            size = df.groupby(hhid).size().rename("household_size")
            base = base.merge(size, left_on=hhid, right_index=True, how="left")

        if head_of_household_only and indid in df.columns:
            head = df[df[indid] == 1]
            # A few households have no member flagged as index 1; for those I
            # just take their first available row so they aren't dropped.
            missing = set(df[hhid]) - set(head[hhid])
            if missing:
                fallback = df[df[hhid].isin(missing)].drop_duplicates(subset=hhid)
                head = pd.concat([head, fallback], ignore_index=True)
            df = head

        df = df.drop_duplicates(subset=hhid)[[hhid] + wanted]
        base = base.merge(df, on=hhid, how="left")

    # ---- Housing (Section I, household level) ---------------------------- #
    specs_i = sections.get("I", [])
    if specs_i:
        sec_i = _read_section(hh_dir, "I")
        wanted = [c for c in (f.column for f in specs_i) if c in sec_i.columns]
        sec_i = sec_i.drop_duplicates(subset=hhid)[[hhid] + wanted]
        base = base.merge(sec_i, on=hhid, how="left")

    # ---- Rename raw columns -> feature names ----------------------------- #
    rename = {f.column: f.name for f in fs.ALL_FEATURES if f.column in base.columns}
    base = base.rename(columns=rename)

    # The rename above catches clustertype -> rural_urban (W4). For W3/W5 the
    # fallback column (y{wave}_rural) isn't in the spec, so rename it here.
    if rural_src and rural_src != "clustertype" and rural_src in base.columns:
        base = base.rename(columns={rural_src: "rural_urban"})

    # Keep the household id around too, for any later joins.
    base = base.rename(columns={hhid: "hhid"})

    # ---- Harmonize per-wave value codes --------------------------------- #
    # W3 water_source / lighting_fuel / rural_urban get remapped to the W4/W5
    # reference scheme here; W4/W5 are identity. See harmonize.py.
    base = harmonize_wave(base, wave)

    # Put the columns in a predictable order: id first, then household_size,
    # then the rest of the features in the order they appear in the spec.
    feature_order = ["hhid", "household_size"] + [
        f.name for f in fs.ALL_FEATURES if f.name != "household_size"
    ]
    cols = [c for c in feature_order if c in base.columns]
    return base[cols].reset_index(drop=True)
