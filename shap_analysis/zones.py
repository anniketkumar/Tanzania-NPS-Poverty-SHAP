"""Map NPS region codes to the paper's 7 administrative zones.

This is my single lookup table for "which zone does a household belong to". I
keep it as one editable dict (mirroring the baseline's `feature_spec.py` pattern)
so that when I need to re-group a region I edit this table, not the analysis code.

WHICH ZONES?
------------
Sende et al. (2025), §V.B, report poverty by **7 zones**:

    Western, Lake, Central, Southern, Northern, Coastal, Zanzibar

Note that **Zanzibar is one of the 7** — it is not an 8th zone. The proposal's
"7 administrative zones" refers to exactly this taxonomy, so matching it is what
makes my zone-stratified SHAP comparable to the paper's zone-level poverty rates.

ASSUMPTION (flagged for verification)
-------------------------------------
The paper does NOT publish an explicit region -> zone table. The mapping below is
a documented, standard NBS zonal grouping of the region codes. Two assignments
are genuinely ambiguous and are the ones to confirm with the paper's authors:

  1. **Southern vs Southern Highlands.** Tanzania's full NBS scheme has a separate
     "Southern Highlands" zone (Iringa, Mbeya, Rukwa, Njombe, Songwe). The paper
     lists only "Southern" among its 7, so I fold the Southern Highlands regions
     into `Southern`. If the paper instead merged them into `Western`, move those
     five codes here.
  2. **Shinyanga (code 17).** Sometimes grouped with `Western`, sometimes with
     `Lake`. I place it in `Lake` (its modern administrative neighbourhood). Flip
     the entry if the paper used the older Western grouping.

Region codes are the standard NBS codes verified against the master data
dictionary: 1-26 are mainland regions, 51-55 are Zanzibar. Codes 22-26 (Njombe,
Katavi, Simiyu, Geita, Songwe) are the post-2012 regions that appear in Waves 4-5.
"""

from __future__ import annotations

# Human-readable region names, kept for documentation / debugging only.
REGION_NAMES: dict[int, str] = {
    1: "Dodoma", 2: "Arusha", 3: "Kilimanjaro", 4: "Tanga", 5: "Morogoro",
    6: "Pwani", 7: "Dar es Salaam", 8: "Lindi", 9: "Mtwara", 10: "Ruvuma",
    11: "Iringa", 12: "Mbeya", 13: "Singida", 14: "Tabora", 15: "Rukwa",
    16: "Kigoma", 17: "Shinyanga", 18: "Kagera", 19: "Mwanza", 20: "Mara",
    21: "Manyara", 22: "Njombe", 23: "Katavi", 24: "Simiyu", 25: "Geita",
    26: "Songwe",
    51: "Kaskazini Unguja", 52: "Kusini Unguja", 53: "Mjini/Magharibi Unguja",
    54: "Kaskazini Pemba", 55: "Kusini Pemba",
}

# The single source of truth: region code -> one of the paper's 7 zones.
REGION_TO_ZONE: dict[int, str] = {
    # Central
    1: "Central", 13: "Central",
    # Northern
    2: "Northern", 3: "Northern", 4: "Northern", 21: "Northern",
    # Coastal (Dar es Salaam + the coastal mainland)
    5: "Coastal", 6: "Coastal", 7: "Coastal",
    # Southern (incl. the Southern Highlands regions 11,12,15,22,26 — see docstring)
    8: "Southern", 9: "Southern", 10: "Southern",
    11: "Southern", 12: "Southern", 15: "Southern", 22: "Southern", 26: "Southern",
    # Western
    14: "Western", 16: "Western", 23: "Western",
    # Lake (Shinyanga=17 placed here — see docstring)
    17: "Lake", 18: "Lake", 19: "Lake", 20: "Lake", 24: "Lake", 25: "Lake",
    # Zanzibar
    51: "Zanzibar", 52: "Zanzibar", 53: "Zanzibar", 54: "Zanzibar", 55: "Zanzibar",
}

# Canonical display order (matches the paper's Table in §V.B, poorest zones first).
ZONE_ORDER: list[str] = [
    "Western", "Lake", "Central", "Southern", "Northern", "Coastal", "Zanzibar",
]

ZANZIBAR_CODES = {51, 52, 53, 54, 55}


def region_code_to_zone(code, include_zanzibar: bool = True) -> str | None:
    """Return the zone name for one region code.

    `code` may be a float (Wave 4 stores region as float) or int; it is coerced
    to int for the lookup. Returns None for unknown codes, or for Zanzibar codes
    when `include_zanzibar` is False (so the caller can drop those households).
    """
    if code is None:
        return None
    try:
        c = int(round(float(code)))
    except (TypeError, ValueError):
        return None
    if not include_zanzibar and c in ZANZIBAR_CODES:
        return None
    return REGION_TO_ZONE.get(c)


def zones_in_use(include_zanzibar: bool = True) -> list[str]:
    """The zone labels expected in the output, in canonical order."""
    if include_zanzibar:
        return list(ZONE_ORDER)
    return [z for z in ZONE_ORDER if z != "Zanzibar"]
