"""Configuration: file paths and tunable model constants.

Everything you might want to tweak about the model lives here so the rest of
the code doesn't have magic numbers scattered through it. A future UI can also
read/write these values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the project (the folder containing this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the raw Kaggle F1 CSVs live. Point this at your "archive2" folder.
DATA_DIR = PROJECT_ROOT / "data"

# Where computed outputs are written.
OUTPUT_DIR = PROJECT_ROOT / "output"

# Expected raw input files (Kaggle "Formula 1 World Championship" dataset).
RACES_CSV = "races.csv"
RESULTS_CSV = "results.csv"
CONSTRUCTORS_CSV = "constructors.csv"
DRIVERS_CSV = "drivers.csv"


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
@dataclass
class EloConfig:
    """Tunable parameters for the ELO model.

    Defaults reproduce the behaviour of the original notebook.
    """

    initial_elo: float = 1000.0

    # K-factors for constructors.
    k_constructor_race: float = 64.0
    k_constructor_quali: float = 16.0

    # K-factors for drivers, split by experience.
    # "Rookie" = race_count <= driver_experience_threshold.
    driver_experience_threshold: int = 30
    k_driver_race_rookie: float = 64.0
    k_driver_race_veteran: float = 32.0
    k_driver_quali_rookie: float = 16.0
    k_driver_quali_veteran: float = 8.0

    # How strongly a driver's result is discounted/boosted by the relative
    # strength of the two constructors involved. Exponent on (rb / ra).
    constructor_adjustment_exponent: float = 2.5

    # Standard ELO logistic scale (the "400" in the classic formula).
    elo_scale: float = 400.0

    # Sentinels in the raw data.
    dnf_sentinel: str = "\\N"   # finishing "position" value for did-not-finish
    no_grid_sentinel: int = 0    # "grid" value meaning no/last qualifying spot


# A ready-to-use default instance.
DEFAULT_CONFIG = EloConfig()
