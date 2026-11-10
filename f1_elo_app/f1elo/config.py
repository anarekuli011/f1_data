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
    k_constructor_race: float = 96.0
    k_constructor_quali: float = 24.0

    # K-factors for drivers, split by experience.
    # "Rookie" = race_count <= driver_experience_threshold.
    driver_experience_threshold: int = 30
    k_driver_race_rookie: float = 64.0
    k_driver_race_veteran: float = 48.0
    k_driver_quali_rookie: float = 16.0
    k_driver_quali_veteran: float = 12.0

    # How strongly a driver's result is discounted/boosted by the relative
    # strength of the two constructors involved. Exponent on (rb / ra).
    constructor_adjustment_exponent: float = 3.0

    # Driver race pass only: when a classified driver is compared with a
    # driver who did not finish (position == dnf_sentinel), both sides of
    # that pair's driver delta are multiplied by this factor -- the DNF
    # driver loses less and the finisher gains correspondingly less, so no
    # rating is created or destroyed. Constructors still take the full
    # penalty (reliability is part of car strength); qualifying is
    # unaffected. 1.0 reproduces the original "DNF = last" behaviour.
    driver_dnf_penalty_factor: float = 0.5

    # Standard ELO logistic scale (the "400" in the classic formula).
    elo_scale: float = 400.0

    # Sentinels in the raw data.
    dnf_sentinel: str = "\\N"   # finishing "position" value for did-not-finish
    no_grid_sentinel: int = 0    # "grid" value meaning no/last qualifying spot


# A ready-to-use default instance.
DEFAULT_CONFIG = EloConfig()
