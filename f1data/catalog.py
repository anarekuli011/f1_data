"""Everything the Jolpica-F1 API offers, in one table.

Each entry says what the endpoint returns, how far back it goes, how many
requests a full download costs, and how useful it is for the ELO engine.
`python cli.py catalog` prints this.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    key: str            # name used on the command line
    path: str           # URL pattern under /ergast/f1/
    since: int          # first season with data
    scope: str          # "global", "season" or "race" (how it is requested)
    rows: str           # rough size of the full history
    requests: str       # rough cost of a full download
    elo: str            # relevance to the ELO engine
    notes: str
    core: bool = False  # part of the default build


CATALOG: list[Endpoint] = [
    Endpoint("seasons", "seasons/", 1950, "global", "~77", "1",
             "Needed", "List of seasons; drives every per-season loop.", core=True),
    Endpoint("circuits", "circuits/", 1950, "global", "~80", "1",
             "Nice to have", "Track name, location, lat/long. Enables per-circuit ELO.", core=True),
    Endpoint("drivers", "drivers/", 1950, "global", "~870", "~9",
             "Needed", "Names, code, number, DOB, nationality.", core=True),
    Endpoint("constructors", "constructors/", 1950, "global", "~215", "~3",
             "Needed", "Team names and nationality.", core=True),
    Endpoint("races", "races/", 1950, "global", "~1,150", "~12",
             "Needed", "Calendar, plus practice/quali/sprint session dates from 2021ish.", core=True),
    Endpoint("results", "{season}/results/", 1950, "season", "~27,000", "~280",
             "Needed", "Grid, finishing position, points, laps, status, time, fastest lap.", core=True),
    Endpoint("qualifying", "{season}/qualifying/", 1994, "season", "~11,000", "~120",
             "Useful", "Actual quali position and Q1/Q2/Q3 times. Better than using race grid "
             "(grid includes penalties).", core=True),
    Endpoint("sprint", "{season}/sprint/", 2021, "season", "~600", "~10",
             "Useful", "Sprint race results. Could be a third ELO pass with its own K-factor.",
             core=True),
    Endpoint("driverstandings", "{season}/{round}/driverstandings/", 1950, "race",
             "~34,000", "~1,150", "Low", "Championship table after each round."),
    Endpoint("constructorstandings", "{season}/{round}/constructorstandings/", 1958, "race",
             "~13,000", "~1,050", "Low", "Constructors' table after each round."),
    Endpoint("pitstops", "{season}/{round}/pitstops/", 2011, "race", "~11,000", "~330",
             "Low", "Lap, stop number, duration of every stop."),
    Endpoint("laps", "{season}/{round}/laps/", 1996, "race", "~600,000", "~7,000",
             "Advanced", "Every driver's time and position on every lap. Enables on-track "
             "battle ELO, but the full history takes about 15 hours at the free rate limit."),
    Endpoint("status", "status/", 1950, "global", "~140", "~2",
             "Useful", "Finishing statuses. From 2025 all retirements/accidents are just "
             "'Retired', so mechanical vs driver DNFs only separate for older seasons."),
]

BY_KEY = {e.key: e for e in CATALOG}
CORE = [e.key for e in CATALOG if e.core]
OPTIONAL = [e.key for e in CATALOG if not e.core]


def print_catalog() -> None:
    header = f"{'endpoint':<23}{'since':<7}{'rows':<10}{'requests':<10}{'ELO value':<14}notes"
    print(header)
    print("-" * len(header) + "-" * 40)
    for e in CATALOG:
        tag = "*" if e.core else " "
        print(f"{tag}{e.key:<22}{e.since:<7}{e.rows:<10}{e.requests:<10}{e.elo:<14}{e.notes}")
    print("\n* = included in the default build.  Rate limit: 4 req/s burst, 500 req/hour.")