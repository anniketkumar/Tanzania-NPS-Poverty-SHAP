"""Build one household-level table by joining the raw survey CSVs together.

The survey splits data across sections (A, B, C, I/J) in separate files. This
module joins them on the household id, keeps just the head-of-household row from
the person-level sections, counts household members to get household_size, and
recovers rural/urban from the wave-appropriate column. What comes out is one row
per household, with columns renamed to the canonical feature names but still
holding the raw survey codes — the one-hot encoding happens later in encode.py.

Wave-specific file paths, column names, and ID columns are configured in
``wave_config.py`` so this loader stays clean of per-wave if/else branches.

History:
  * Week 4 (2026-07-14): W3 support via harmonize.py + rural_urban recovery.
  * Week 5-6 (2026-07-24): W1/W2 support via wave_config.py + extended harmonize.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import feature_spec as fs
from .harmonize import harmonize_wave
from .wave_config import get_wave_config, WaveConfig


def _household_dir(converted_data_dir: Path, wave: int) -> Path:
    d = Path(converted_data_dir) / f"wave{wave}" / "household"
    if not d.is_dir():
        raise FileNotFoundError(
            f"Expected household CSVs at {d}. Check config.data.converted_data_dir "
            f"and that wave {wave} has been converted."
        )
    return d


def _read_section(hh_dir: Path, section: str, wc: WaveConfig) -> pd.DataFrame:
    """Read a section CSV, using the wave-specific filename from WaveConfig."""
    filename = wc.section_files[section]
    path = hh_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing section file: {path}")
    # utf-8-sig strips the byte-order mark some of these converted CSVs start
    # with; low_memory=False stops pandas complaining about mixed-type columns.
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def _resolve_column(canonical: str, wc: WaveConfig) -> str | None:
    """Map a canonical W4 column name to the actual name in this wave.

    Returns ``None`` if the feature does not exist in this wave (e.g.
    hh_i15 / child_stool_disposal in W1/W2).
    Returns the canonical name unchanged for W3–5 (empty column_map).
    """
    if not wc.column_map:
        # W3–5: identity mapping
        return canonical
    return wc.column_map.get(canonical, canonical)


def _actual_columns_for_section(
    specs: list[fs.FeatureSpec], wc: WaveConfig, df_columns: list[str]
) -> list[str]:
    """Return the list of actual column names present in `df` for a section's specs."""
    result = []
    for spec in specs:
        actual = _resolve_column(spec.column, wc)
        if actual is not None and actual in df_columns:
            result.append(actual)
    return result


def _build_rename_map(wc: WaveConfig, present_columns: set[str]) -> dict[str, str]:
    """Build {actual_column → feature_name} for all features present in the data.

    For W3–5 (empty column_map) this is just {spec.column: spec.name}.
    For W1/W2 it maps the wave-specific column name to the feature name.
    """
    rename = {}
    for spec in fs.ALL_FEATURES:
        actual = _resolve_column(spec.column, wc)
        if actual is not None and actual in present_columns:
            rename[actual] = spec.name
    return rename


# Cache of already-read DataFrames per physical file, to avoid reading the W1
# merged file twice (once for "B" and once for "C").
_section_cache: dict[str, pd.DataFrame] = {}


def load_wave_features(
    converted_data_dir: Path,
    wave: int,
    head_of_household_only: bool = True,
) -> pd.DataFrame:
    """Return a one-row-per-household DataFrame of raw (un-encoded) features."""
    hh_dir = _household_dir(converted_data_dir, wave)
    wc = get_wave_config(wave)
    hhid = wc.hhid
    indid = wc.indid

    sections = fs.features_by_section()

    # Clear the section cache at the start of each wave load.
    _section_cache.clear()

    def _read_cached(section: str) -> pd.DataFrame:
        """Read a section, caching by physical filename to avoid double reads."""
        filename = wc.section_files[section]
        if filename not in _section_cache:
            _section_cache[filename] = _read_section(hh_dir, section, wc)
        return _section_cache[filename]

    # ---- Section A (household level) ------------------------------------- #
    sec_a = _read_cached("A")
    a_specs = sections.get("A", [])
    a_wanted = _actual_columns_for_section(a_specs, wc, list(sec_a.columns))
    a_cols = [hhid] + a_wanted

    # rural/urban recovery: use the wave_config's dedicated rural_src when set
    # (W1: locality, W2: y2_rural, W3: y3_rural). For W4 the existing
    # clustertype logic applies. For W5, y5_rural.
    rural_src = None
    if wc.rural_src and wc.rural_src in sec_a.columns and sec_a[wc.rural_src].notna().any():
        rural_src = wc.rural_src
    elif "clustertype" in sec_a.columns and sec_a["clustertype"].notna().any():
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
        df = _read_cached(sec_name)
        wanted = _actual_columns_for_section(specs, wc, list(df.columns))

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

    # ---- Housing (Section I/J, household level) -------------------------- #
    specs_i = sections.get("I", [])
    if specs_i:
        sec_i = _read_cached("I")
        wanted = _actual_columns_for_section(specs_i, wc, list(sec_i.columns))
        sec_i = sec_i.drop_duplicates(subset=hhid)[[hhid] + wanted]
        base = base.merge(sec_i, on=hhid, how="left")

    # ---- Rename raw columns -> feature names ----------------------------- #
    rename = _build_rename_map(wc, set(base.columns))
    base = base.rename(columns=rename)

    # The rename above catches clustertype -> rural_urban (W4). For other waves
    # the rural_src column isn't in the spec, so rename it here.
    if rural_src and rural_src != "clustertype" and rural_src in base.columns:
        base = base.rename(columns={rural_src: "rural_urban"})

    # Keep the household id around too, for any later joins.
    base = base.rename(columns={hhid: "hhid"})

    # ---- Harmonize per-wave value codes --------------------------------- #
    # W1/W2/W3 value codes get remapped to the W4/W5 reference scheme here;
    # W4/W5 are identity. See harmonize.py.
    base = harmonize_wave(base, wave)

    # Put the columns in a predictable order: id first, then household_size,
    # then the rest of the features in the order they appear in the spec.
    feature_order = ["hhid", "household_size"] + [
        f.name for f in fs.ALL_FEATURES if f.name != "household_size"
    ]
    cols = [c for c in feature_order if c in base.columns]
    return base[cols].reset_index(drop=True)
