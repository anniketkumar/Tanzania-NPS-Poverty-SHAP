"""Per-wave configuration: file names, column mappings, and ID columns.

Waves 3–5 share the same HH_SEC_{A,B,C,I}.csv layout with hh_b*/hh_c*/hh_i*
column prefixes. Waves 1 and 2 deviate:

  * **Wave 1 (2008/09):** merged files with Swahili-prefix columns (sbq*, scq*,
    sjq*). Housing lives in Section J, not I. The household file is
    ``SEC_B_C_D_E1_F_G1_U.csv`` (individuals) and
    ``SEC_H1_J_K2_O2_P1_Q1_S1.csv`` (household-level housing). Section A is
    ``SEC_A_T.csv``. The HH ID column is plain ``hhid``; the individual-member
    column is ``sbmemno``.

  * **Wave 2 (2010/11):** uses the HH_SEC_*.csv naming convention like W3–5,
    but housing is in ``HH_SEC_J1.csv`` with ``hh_j*`` columns instead of
    ``HH_SEC_I.csv`` / ``hh_i*``. Section I in W2 is food security / shocks,
    not housing. ``child_stool_disposal`` (hh_i15) does not exist.

This module provides ``get_wave_config(wave)`` which returns everything the
loader needs to read and rename columns for any wave.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WaveConfig:
    """Everything the loader needs to adapt to a specific wave."""

    wave: int
    hhid: str                          # household-id column name
    indid: str                         # individual-member-id column name

    # Map from logical section label ("A", "B", "C", "I") to the CSV filename
    # sitting in ``converted data/wave{N}/household/``.
    # For W1, B and C point to the same merged file — the loader handles that.
    section_files: dict[str, str]

    # Map from the canonical W4 column name (used in feature_spec.py) to the
    # actual column name in *this* wave's CSV files.  Keys that map to ``None``
    # indicate a feature that does not exist in this wave (e.g. hh_i15 /
    # child_stool_disposal in W1/W2).  W3–5 use identity (empty dict → no
    # renaming needed).
    column_map: dict[str, str | None]

    # The column from which rural/urban is sourced in this wave.
    # ``None`` means the loader should use its existing fallback logic.
    rural_src: str | None = None

    # Whether Sections B and C live in the same physical CSV file.
    # True only for Wave 1's merged SEC_B_C_D_E1_F_G1_U.csv.
    bc_merged: bool = False


# ------------------------------------------------------------------ #
# Wave 1 (2008/09)                                                    #
# ------------------------------------------------------------------ #
_W1 = WaveConfig(
    wave=1,
    hhid="hhid",
    indid="sbmemno",
    section_files={
        "A": "SEC_A_T.csv",
        "B": "SEC_B_C_D_E1_F_G1_U.csv",
        "C": "SEC_B_C_D_E1_F_G1_U.csv",   # same physical file as B
        "I": "SEC_H1_J_K2_O2_P1_Q1_S1.csv",  # Section J (housing)
    },
    column_map={
        # --- Demographics (Section B) ---
        "hh_b02": "sbq2",       # head_sex
        "hh_b04": "sbq4",       # head_age
        "hh_b05": "sbq5",       # relationship_to_head
        # --- Education (Section C) ---
        "hh_c01": "scq1",       # age_5_or_above
        "hh_c07": "scq6",       # education_level (highest completed)
        "hh_c09": "scq7",       # grade_currently_attending
        # --- Housing (Section J in W1) ---
        "hh_i01": "sjq1",       # housing_tenure
        "hh_i09": "sjq3_1",     # rooms
        "hh_i10": "sjq6",       # floor_material
        "hh_i11": "sjq15",      # garbage_disposal
        "hh_i12": "sjq16",      # toilet_facility
        "hh_i15": None,         # child_stool_disposal — NOT IN W1
        "hh_i17": "sjq18",      # lighting_fuel
        "hh_i18": "sjq19",      # electricity_source
        "hh_i19": "sjq8",       # water_source
        # --- Geographic (Section A) ---
        "hh_a01_1": "region",   # region code
        "clustertype": None,    # no clustertype; rural_src handles it
    },
    rural_src="locality",       # 1=Rural, 2=Urban, 3=Mixture
    bc_merged=True,
)


# ------------------------------------------------------------------ #
# Wave 2 (2010/11)                                                    #
# ------------------------------------------------------------------ #
_W2 = WaveConfig(
    wave=2,
    hhid="y2_hhid",
    indid="indidy2",
    section_files={
        "A": "HH_SEC_A.csv",
        "B": "HH_SEC_B.csv",
        "C": "HH_SEC_C.csv",
        "I": "HH_SEC_J1.csv",  # Section J (housing), not I
    },
    column_map={
        # --- Demographics (Section B) — same names as W3-5 ---
        "hh_b02": "hh_b02",
        "hh_b04": "hh_b04",
        "hh_b05": "hh_b05",
        # --- Education (Section C) — same names as W3-5 ---
        "hh_c01": "hh_c01",
        "hh_c07": "hh_c07",
        "hh_c09": "hh_c09",
        # --- Housing (Section J in W2, hh_j* prefix) ---
        "hh_i01": "hh_j01",    # housing_tenure
        "hh_i09": "hh_j04_1",  # rooms
        "hh_i10": "hh_j07",    # floor_material
        "hh_i11": "hh_j09",    # garbage_disposal
        "hh_i12": "hh_j10",    # toilet_facility
        "hh_i15": None,         # child_stool_disposal — NOT IN W2
        "hh_i17": "hh_j17",    # lighting_fuel
        "hh_i18": "hh_j18",    # electricity_source
        "hh_i19": "hh_j19",    # water_source
        # --- Geographic (Section A) ---
        "hh_a01_1": "region",   # no value labels in dict but codes are 1-21,51-55
        "clustertype": None,    # no clustertype; y2_rural used via rural_src
    },
    rural_src="y2_rural",       # 0=urban, 1=rural (same scheme as W3)
)


# ------------------------------------------------------------------ #
# Waves 3–5 — identity (existing loader logic handles these)          #
# ------------------------------------------------------------------ #
_W3 = WaveConfig(wave=3, hhid="y3_hhid", indid="indidy3",
                 section_files={"A": "HH_SEC_A.csv", "B": "HH_SEC_B.csv",
                                "C": "HH_SEC_C.csv", "I": "HH_SEC_I.csv"},
                 column_map={})

_W4 = WaveConfig(wave=4, hhid="y4_hhid", indid="indidy4",
                 section_files={"A": "HH_SEC_A.csv", "B": "HH_SEC_B.csv",
                                "C": "HH_SEC_C.csv", "I": "HH_SEC_I.csv"},
                 column_map={})

_W5 = WaveConfig(wave=5, hhid="y5_hhid", indid="indidy5",
                 section_files={"A": "HH_SEC_A.csv", "B": "HH_SEC_B.csv",
                                "C": "HH_SEC_C.csv", "I": "HH_SEC_I.csv"},
                 column_map={})


_CONFIGS = {c.wave: c for c in (_W1, _W2, _W3, _W4, _W5)}


def get_wave_config(wave: int) -> WaveConfig:
    """Return the configuration for a given wave number.

    Raises ``KeyError`` if the wave is unknown.
    """
    try:
        return _CONFIGS[wave]
    except KeyError:
        raise KeyError(
            f"No configuration for wave {wave}. "
            f"Known waves: {sorted(_CONFIGS)}"
        ) from None
