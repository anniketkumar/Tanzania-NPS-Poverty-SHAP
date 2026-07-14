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
    # hh_b05 is relationship-to-head (1=HEAD, 2=SPOUSE, ...), NOT marital status;
    # renamed 2026-07-14 (Week 4). Codes are identical across W2-W5, so no recode.
    FeatureSpec("relationship_to_head", "B", "hh_b05", "categorical"),
]

# --------------------------------------------------------------------------- #
# Education (Section C) — head row.                                            #
# --------------------------------------------------------------------------- #
EDUCATION = [
    # hh_c01 is "Is [NAME] 5 years or above?" (1=YES, 2=NO), NOT school
    # attendance; renamed 2026-07-14 (Week 4). Codes identical across waves.
    FeatureSpec("age_5_or_above", "C", "hh_c01", "categorical"),
    FeatureSpec("education_level", "C", "hh_c07", "categorical",
                "W5 recodes these values — remap before cross-wave use. "
                "W3 == W4, so no W3 recode needed.", verify=True),
    # hh_c09 is "What grade is [NAME] currently attending?", NOT literacy;
    # renamed 2026-07-14 (Week 4). W3 == W4 codes, so no W3 recode.
    FeatureSpec("grade_currently_attending", "C", "hh_c09", "categorical", verify=True),
]

# --------------------------------------------------------------------------- #
# Living standards / housing (Section I in W3-5).                              #
# --------------------------------------------------------------------------- #
HOUSING = [
    FeatureSpec("housing_tenure", "I", "hh_i01", "categorical"),
    FeatureSpec("rooms", "I", "hh_i09", "numeric"),
    FeatureSpec("floor_material", "I", "hh_i10", "categorical",
                "W5 expanded codes (3 -> 6) — collapse before cross-wave use", verify=True),
    FeatureSpec("garbage_disposal", "I", "hh_i11", "categorical",
                "W3 == W4 (6 codes); W5 silently expanded to 11 — locked W5 issue"),
    FeatureSpec("toilet_facility", "I", "hh_i12", "categorical"),
    FeatureSpec("child_stool_disposal", "I", "hh_i15", "categorical"),
    # hh_i17 codes 9/10 are SWAPPED between W3 (9=OTHER, 10=TORCH) and W4/W5
    # (9=TORCH, 10=OTHER). Was NOT flagged; W3 gets a 9<->10 swap in harmonize.py.
    FeatureSpec("lighting_fuel", "I", "hh_i17", "categorical", verify=True),
    FeatureSpec("electricity_source", "I", "hh_i18", "categorical",
                "structurally missing (~30-62%; W3 is worst at 72.6%); relies on KNN imputation"),
    # W3 uses a 14-category access taxonomy vs W4/W5's 12-category source taxonomy.
    # harmonize.py remaps W3 -> the W4/W5 reference scheme (crosswalk approved
    # 2026-07-14; see Week4_Harmonization_Crosswalk.md).
    FeatureSpec("water_source", "I", "hh_i19", "categorical",
                "W3 remapped to W4/W5 scheme in harmonize.py", verify=True),
]

# --------------------------------------------------------------------------- #
# Socioeconomic / geographic (Section A) — household level.                   #
# --------------------------------------------------------------------------- #
GEOGRAPHIC = [
    FeatureSpec("region", "A", "hh_a01_1", "categorical", "region code"),
    # rural/urban is recovered specially in the loader with a per-wave fallback:
    #   W4 -> clustertype (1=RURAL, 2=URBAN); its y4_rural is 100% missing.
    #   W3 -> y3_rural (0=URBAN, 1=RURAL) -- canonicalized to 1=RURAL,2=URBAN.
    #   W5 -> y5_rural (1=RURAL, 2=URBAN).
    # The `column` here is the W4 source; the loader adds the W3/W5 fallback and
    # harmonize.py normalizes the codes. (Before Week 4 this was dropped in W3/W5.)
    FeatureSpec("rural_urban", "A", "clustertype", "categorical",
                "loader fallback to y{wave}_rural; codes normalized in harmonize.py"),
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
