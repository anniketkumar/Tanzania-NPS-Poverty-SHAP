"""The list of features and which raw CSV column each one comes from.

This is my single lookup table for "what turns into a feature". I kept it as a
list of `FeatureSpec` records on purpose: when I need to add, drop, or repoint a
feature, I edit this table instead of touching the loader code.

Each entry maps to a real column I confirmed in the wave-4/5 household files
(HH_SEC_A/B/C/I.csv). Waves 4 and 5 share the same hh_b*/hh_c*/hh_i* column
names, so one table covers both; only the id columns (y4_/y5_ and
indidy4/indidy5) differ, and those get swapped in when the data is loaded.

`verify=True` marks features whose value codes changed between waves. They're
fine for a single-wave run (which is what I'm doing), but I'd need to remap
their codes before mixing waves together.
"""

from __future__ import annotations

from dataclasses import dataclass


# "categorical" -> one-hot encoded; "numeric" -> kept as-is (then scaled).
@dataclass(frozen=True)
class FeatureSpec:
    name: str          # human-readable feature name (becomes column prefix)
    section: str       # which HH_SEC_*.csv file, e.g. "B", "C", "I", "A"
    column: str        # raw column name in that section's CSV
    kind: str          # "categorical" | "numeric"
    note: str = ""
    verify: bool = False


# --------------------------------------------------------------------------- #
# Demographics (Section B) — keyed at the individual level; head row is kept.  #
# --------------------------------------------------------------------------- #
DEMOGRAPHICS = [
    FeatureSpec("head_sex", "B", "hh_b02", "categorical", "1=Male, 2=Female"),
    FeatureSpec("head_age", "B", "hh_b04", "numeric", "age in years"),
    FeatureSpec("marital_status", "B", "hh_b05", "categorical", verify=True),
]

# --------------------------------------------------------------------------- #
# Education (Section C) — head row.                                            #
# --------------------------------------------------------------------------- #
EDUCATION = [
    FeatureSpec("school_attendance", "C", "hh_c01", "categorical"),
    FeatureSpec("education_level", "C", "hh_c07", "categorical",
                "W5 recodes these values — remap before cross-wave use", verify=True),
    FeatureSpec("literacy", "C", "hh_c09", "categorical", verify=True),
]

# --------------------------------------------------------------------------- #
# Living standards / housing (Section I in W3-5).                              #
# --------------------------------------------------------------------------- #
HOUSING = [
    FeatureSpec("housing_tenure", "I", "hh_i01", "categorical"),
    FeatureSpec("rooms", "I", "hh_i09", "numeric"),
    FeatureSpec("floor_material", "I", "hh_i10", "categorical",
                "W5 expanded codes (3 -> 6) — collapse before cross-wave use", verify=True),
    FeatureSpec("garbage_disposal", "I", "hh_i11", "categorical"),
    FeatureSpec("toilet_facility", "I", "hh_i12", "categorical"),
    FeatureSpec("child_stool_disposal", "I", "hh_i15", "categorical"),
    FeatureSpec("lighting_fuel", "I", "hh_i17", "categorical"),
    FeatureSpec("electricity_source", "I", "hh_i18", "categorical",
                "structurally missing (~30-62%); relies on KNN imputation"),
    FeatureSpec("water_source", "I", "hh_i19", "categorical",
                "W3 uses finer codes than W4-5 — remap before cross-wave use", verify=True),
]

# --------------------------------------------------------------------------- #
# Socioeconomic / geographic (Section A) — household level.                   #
# --------------------------------------------------------------------------- #
GEOGRAPHIC = [
    FeatureSpec("region", "A", "hh_a01_1", "categorical", "region code"),
    # rural/urban: in W4 the dedicated y4_rural column is 100% missing; recover
    # from clustertype (1=RURAL, 2=URBAN). Handled specially in the loader.
    FeatureSpec("rural_urban", "A", "clustertype", "categorical",
                "W4: y4_rural is 100% missing; clustertype is the fallback"),
]

# household_size is computed (count of Section B members per household), not read
# from a column. The loader injects it as a numeric feature.
COMPUTED_NUMERIC = ["household_size"]


ALL_FEATURES: list[FeatureSpec] = DEMOGRAPHICS + EDUCATION + HOUSING + GEOGRAPHIC


def categorical_feature_names() -> list[str]:
    return [f.name for f in ALL_FEATURES if f.kind == "categorical"]


def numeric_feature_names() -> list[str]:
    return [f.name for f in ALL_FEATURES if f.kind == "numeric"] + COMPUTED_NUMERIC


def features_by_section() -> dict[str, list[FeatureSpec]]:
    out: dict[str, list[FeatureSpec]] = {}
    for f in ALL_FEATURES:
        out.setdefault(f.section, []).append(f)
    return out
