"""Per-wave value-code harmonization, applied after the loader renames columns.

Waves share the same `hh_i*`/`hh_b*`/`hh_c*` column *names*, but a few features
use different *value codes* across waves. This module remaps those codes so a
feature means the same thing in every wave.

Scope (approved 2026-07-14, see `Week4_Harmonization_Crosswalk.md`):
  * Only **Wave 3** needs value recodes; Waves 4/5 are the reference scheme and
    are left untouched (identity), so their locked results do not change.
  * Three features are affected in W3: `water_source` (14-cat access taxonomy ->
    W4/W5 12-cat source taxonomy), `lighting_fuel` (codes 9<->10 swapped), and
    `rural_urban` (W3's y3_rural uses 0=URBAN/1=RURAL; canonicalized to the
    W4/W5 convention 1=RURAL, 2=URBAN).

The maps are keyed by the *feature name* (post-rename), not the raw column.
Codes not present in a map pass through unchanged (identity), so partial maps
like the lighting_fuel swap only touch the codes they list.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- #
# Wave 3 crosswalks                                                            #
# --------------------------------------------------------------------------- #

# water_source hh_i19: W3 (14 categories) -> W4/W5 (12 categories).
# Rationale per code is documented in Week4_Harmonization_Crosswalk.md.
#   1 piped-inside, 2 private standpipe, 3 public standpipe, 4 neighbour,
#   6 subsidized vending station  -> 1 PIPED WATER
#   5 water vendor                -> 9 CART WITH SMALL TANK/DRUM
#   7 truck/tanker                -> 10 TANKER-TRUCK
#   8 protected well + pump       -> 2 TUBEWELL/BOREHOLE
#   9 unprotected well + pump, 11 unprotected well no-pump -> 4 UNPROTECTED DUGWELL
#   10 protected well no-pump     -> 3 PROTECTED DUG WELL
#   12 river/lake/spring/pond     -> 11 SURFACE WATER
#   13 rainwater                  -> 7 RAINWATER COLLECTION
#   14 other                      -> 12 OTHER
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
# Swap W3's 9 and 10 onto the W4/W5 code positions. Codes 1-8 are identical.
_LIGHTING_FUEL_W3 = {9: 10, 10: 9}

# rural_urban: canonicalize W3's y3_rural (0=URBAN, 1=RURAL) to the W4/W5
# convention (1=RURAL, 2=URBAN). W4 (clustertype) and W5 (y5_rural) already use
# 1=RURAL/2=URBAN, so they need no remap.
_RURAL_URBAN_W3 = {0: 2, 1: 1}

# feature name -> {raw code: harmonized code}, per wave. Empty dict = identity.
RECODES: dict[int, dict[str, dict[int, int]]] = {
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
