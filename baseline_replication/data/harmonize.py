"""Per-wave value-code harmonization, applied after the loader renames columns.

Waves share the same feature *names* (post-rename), but use different *value
codes* across waves. This module remaps those codes so a feature means the same
thing in every wave.

All recodes map onto the **Wave 4 reference scheme** — the locked standard.
Waves 4 and 5 are identity (no recode needed); their locked results do not
change.

Scope:
  * Wave 3 (approved 2026-07-14, Week4_Harmonization_Crosswalk.md):
    water_source (14→12 cat), lighting_fuel (9↔10 swap), rural_urban (0/1→1/2).

  * Wave 2 (approved 2026-07-24, Week5_6 crosswalk):
    toilet_facility (8-cat reshuffled), lighting_fuel (9→10),
    water_source (14→12, same as W3 crosswalk), rural_urban (0/1→1/2),
    education_level + grade_currently_attending (PREFORM 1 code shift).

  * Wave 1 (approved 2026-07-24, Week5_6 crosswalk):
    toilet_facility (5-cat→8-cat), lighting_fuel (9→10),
    water_source (11-cat→12-cat, new crosswalk), rural_urban (1/2/3→1/2).

The maps are keyed by the *feature name* (post-rename), not the raw column.
Codes not present in a map pass through unchanged (identity), so partial maps
like the lighting_fuel swap only touch the codes they list.
"""

from __future__ import annotations

import pandas as pd

# =========================================================================== #
# Wave 3 crosswalks (unchanged from Week 4)                                   #
# =========================================================================== #

# water_source hh_i19: W3 (14 categories) -> W4/W5 (12 categories).
# Rationale per code is documented in Week4_Harmonization_Crosswalk.md.
_WATER_SOURCE_W3 = {
    1: 1, 2: 1, 3: 1, 4: 1, 6: 1,
    5: 9,
    7: 10,
    8: 2,
    9: 4, 11: 4,
    10: 3,
    12: 11,
    13: 7,
    14: 12,
}

# lighting_fuel hh_i17: W3 has 9=OTHER, 10=TORCH; W4/W5 have 9=TORCH, 10=OTHER.
_LIGHTING_FUEL_W3 = {9: 10, 10: 9}

# rural_urban: W3's y3_rural uses 0=URBAN, 1=RURAL; canonicalize to W4/W5.
_RURAL_URBAN_W3 = {0: 2, 1: 1}


# =========================================================================== #
# Wave 2 crosswalks (approved 2026-07-24)                                      #
# =========================================================================== #

# toilet_facility hh_j10: W2's 8 codes are almost completely reshuffled vs W4.
# W2 code 1 = NO TOILET but W4 code 1 = FLUSH TOILET — would be silently wrong
# without this crosswalk.
#   W2: 1=NO TOILET, 2=FLUSH, 3=POUR FLUSH, 4=VIP, 5=ECOSAN,
#       6=UNIMPROVED PIT (slab not washable), 7=IMPROVED PIT (slab washable), 8=OTHER
#   W4: 1=FLUSH, 2=OPEN PIT, 3=POUR FLUSH, 4=ECOSAN, 5=VIP,
#       6=IMPROVED PIT (slab washable), 7=NO TOILET, 8=OTHER
_TOILET_FACILITY_W2 = {
    1: 7,   # NO TOILET -> W4 code 7
    2: 1,   # FLUSH TOILET -> W4 code 1
    3: 3,   # POUR FLUSH -> W4 code 3 (identity)
    4: 5,   # VIP -> W4 code 5
    5: 4,   # ECOSAN -> W4 code 4
    6: 2,   # UNIMPROVED PIT LATRINE -> W4 code 2 (PIT LATRINE WITH SLAB/OPEN PIT)
    7: 6,   # IMPROVED PIT LATRINE -> W4 code 6
    8: 8,   # OTHER -> W4 code 8 (identity)
}

# lighting_fuel hh_j17: W2 has only 9 codes. Code 9 = OTHER in W2, but
# code 9 = TORCH in W4. W2 has no TORCH category. Remap: 9 -> 10 (OTHER).
_LIGHTING_FUEL_W2 = {9: 10}

# water_source hh_j19: W2 uses the SAME 14-category scheme as W3.
# Reuse the W3 crosswalk exactly.
_WATER_SOURCE_W2 = _WATER_SOURCE_W3.copy()

# rural_urban y2_rural: 0=urban, 1=rural — same as W3.
_RURAL_URBAN_W2 = {0: 2, 1: 1}

# education_level and grade_currently_attending hh_c07/hh_c09:
# W2 inserted PREFORM 1 at code 19, pushing MS+COURSE to code 20.
# W4 uses 19 = MS+COURSE, has no code 20.
# Crosswalk: collapse both into 19 (MS+COURSE in W4 scheme).
_EDUCATION_LEVEL_W2 = {19: 19, 20: 19}   # PREFORM 1 -> MS+COURSE; MS+COURSE -> MS+COURSE
_GRADE_W2 = _EDUCATION_LEVEL_W2.copy()


# =========================================================================== #
# Wave 1 crosswalks (approved 2026-07-24)                                      #
# =========================================================================== #

# toilet_facility sjq16: W1 has only 5 coarse categories.
#   W1: 1=Flush, 2=VIP, 3=Pit Latrine, 4=Other, 5=No Toilet
#   W4: 1=FLUSH, 2=OPEN PIT, 3=POUR FLUSH, 4=ECOSAN, 5=VIP,
#       6=IMPROVED PIT, 7=NO TOILET, 8=OTHER
_TOILET_FACILITY_W1 = {
    1: 1,   # Flush Toilet -> W4 code 1 (FLUSH TOILET)
    2: 5,   # VIP -> W4 code 5 (VIP LATRINE)
    3: 2,   # Pit Latrine -> W4 code 2 (PIT LATRINE WITH SLAB/OPEN PIT)
    4: 8,   # Other -> W4 code 8 (OTHER)
    5: 7,   # No Toilet -> W4 code 7 (NO TOILET)
}

# lighting_fuel sjq18: same 9-code scheme as W2 — code 9 = OTHER, no TORCH.
_LIGHTING_FUEL_W1 = {9: 10}

# water_source sjq8: W1 has 11 categories (Swahili labels).
#   1 piped inside, 2 piped outside, 3 community piped, 4 piped from neighbour
#       -> all map to W4 code 1 (PIPED WATER)
#   5 water vendor -> W4 code 9 (CART WITH SMALL TANK)
#   6 tanker delivery -> W4 code 10 (TANKER-TRUCK)
#   7 well with pump -> W4 code 2 (TUBEWELL/BOREHOLE)
#   8 well without pump -> W4 code 4 (UNPROTECTED DUGWELL) — judgement call,
#     W1 doesn't distinguish protected/unprotected for wells without pumps
#   9 river/lake/spring/pond -> W4 code 11 (SURFACE WATER)
#   10 rainwater -> W4 code 7 (RAINWATER COLLECTION)
#   11 other -> W4 code 12 (OTHER)
_WATER_SOURCE_W1 = {
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 9,
    6: 10,
    7: 2,
    8: 4,
    9: 11,
    10: 7,
    11: 12,
}

# rural_urban (locality): W1 uses 1=Rural, 2=Urban, 3=Mixture.
# Map Mixture -> Urban (recommended in crosswalk).
_RURAL_URBAN_W1 = {1: 1, 2: 2, 3: 2}


# =========================================================================== #
# Master recode table                                                          #
# =========================================================================== #

# feature name -> {raw code: harmonized code}, per wave. Empty dict = identity.
RECODES: dict[int, dict[str, dict[int, int]]] = {
    1: {
        "water_source": _WATER_SOURCE_W1,
        "lighting_fuel": _LIGHTING_FUEL_W1,
        "toilet_facility": _TOILET_FACILITY_W1,
        "rural_urban": _RURAL_URBAN_W1,
    },
    2: {
        "water_source": _WATER_SOURCE_W2,
        "lighting_fuel": _LIGHTING_FUEL_W2,
        "toilet_facility": _TOILET_FACILITY_W2,
        "rural_urban": _RURAL_URBAN_W2,
        "education_level": _EDUCATION_LEVEL_W2,
        "grade_currently_attending": _GRADE_W2,
    },
    3: {
        "water_source": _WATER_SOURCE_W3,
        "lighting_fuel": _LIGHTING_FUEL_W3,
        "rural_urban": _RURAL_URBAN_W3,
    },
    4: {},
    5: {},
}


def _remap_value(v, code_map: dict[int, int]):
    """Map one raw code through `code_map`, leaving NaN and unlisted codes as-is."""
    if pd.isna(v):
        return v
    try:
        key = int(float(v))
    except (ValueError, TypeError):
        return v
    return code_map.get(key, key)


def harmonize_wave(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    """Return `df` with this wave's value codes remapped in place (copy-safe).

    Only the features listed for `wave` in RECODES are touched; every other
    column is returned unchanged. Waves 4 and 5 are identity by design.
    """
    code_maps = RECODES.get(wave, {})
    if not code_maps:
        return df
    df = df.copy()
    for feature, code_map in code_maps.items():
        if feature in df.columns:
            df[feature] = df[feature].map(lambda v: _remap_value(v, code_map))
    return df
