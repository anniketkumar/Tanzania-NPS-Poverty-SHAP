"""Data assembly: raw NPS CSVs -> head-of-household feature table -> design matrix."""

from .loader import load_wave_features
from .encode import build_design_matrix

__all__ = ["load_wave_features", "build_design_matrix"]
