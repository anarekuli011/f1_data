"""Queries and aggregations over the computed ratings.

These functions take the engine output (the race frame with EloRating /
DriverEloRating filled in) plus the constructors/drivers lookup tables, and
return tidy DataFrames. A UI can render these directly.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------
def _with_driver_names(df: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left")
    out["driverName"] = out["forename"].fillna("") + " " + out["surname"].fillna("")
    out["driverName"] = out["driverName"].str.strip()
    return out


def _with_constructor_names(df: pd.DataFrame, constructors: pd.DataFrame) -> pd.DataFrame:
    return df.merge(constructors[["constructorId", "name"]], on="constructorId", how="left")


# ---------------------------------------------------------------------------
# Peak-rating leaderboards
# ---------------------------------------------------------------------------
def top_constructors_by_peak(results: pd.DataFrame, constructors: pd.DataFrame,
                             n: int = 20) -> pd.DataFrame:
    """Highest ELO any constructor ever reached, with the race it happened at."""
    idx = results.groupby("constructorId")["EloRating"].idxmax()
    peak = results.loc[idx, ["constructorId", "EloRating", "date", "name"]].copy()
    peak = _with_constructor_names(peak, constructors)
    peak["EloRating"] = peak["EloRating"].round().astype(int)
    peak = peak.sort_values("EloRating", ascending=False).head(n).reset_index(drop=True)
    peak.index += 1
    return peak.rename(columns={"name_x": "race", "name_y": "constructor"})


def top_drivers_by_peak(results: pd.DataFrame, drivers: pd.DataFrame,
                        n: int = 50, max_date: str | None = None) -> pd.DataFrame:
    """Highest ELO any driver ever reached (optionally up to a cutoff date)."""
    df = results
    if max_date is not None:
        df = df[df["date"] <= pd.to_datetime(max_date)]
    idx = df.groupby("driverId")["DriverEloRating"].idxmax()
    peak = df.loc[idx, ["driverId", "DriverEloRating", "date", "name"]].copy()
    peak = _with_driver_names(peak, drivers)
    peak["DriverEloRating"] = peak["DriverEloRating"].round().astype(int)
    peak = peak.sort_values("DriverEloRating", ascending=False).head(n).reset_index(drop=True)
    peak.index += 1
    return peak[["driverName", "DriverEloRating", "date", "name"]].rename(
        columns={"name": "race"}
    )


# ---------------------------------------------------------------------------
# Current / last-known ratings
# ---------------------------------------------------------------------------
def current_constructor_ratings(engine_constructor_elo: dict, constructors: pd.DataFrame,
                                names: list[str] | None = None) -> pd.DataFrame:
    """Final ELO for each constructor, optionally filtered to a name list."""
    elo = pd.DataFrame(engine_constructor_elo.items(),
                       columns=["constructorId", "EloRating"])
    out = constructors.merge(elo, on="constructorId", how="left")
    if names is not None:
        out = out[out["name"].isin(names)]
    out = out.dropna(subset=["EloRating"]).copy()
    out["EloRating"] = out["EloRating"].round().astype(int)
    return out.sort_values("EloRating", ascending=False)[["name", "EloRating"]] \
        .reset_index(drop=True)


def current_driver_ratings(engine_driver_elo: dict, drivers: pd.DataFrame,
                           names: list[str] | None = None) -> pd.DataFrame:
    """Final ELO for each driver, optionally filtered to a 'First Last' list."""
    elo = pd.DataFrame(engine_driver_elo.items(),
                       columns=["driverId", "EloRating"])
    out = drivers.merge(elo, on="driverId", how="left")
    out["driverName"] = (out["forename"].fillna("") + " " + out["surname"].fillna("")).str.strip()
    if names is not None:
        out = out[out["driverName"].isin(names)]
    out = out.dropna(subset=["EloRating"]).copy()
    out["EloRating"] = out["EloRating"].round().astype(int)
    return out.sort_values("EloRating", ascending=False)[["driverName", "EloRating"]] \
        .reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rating trajectories (for line charts)
# ---------------------------------------------------------------------------
def driver_rating_history(results: pd.DataFrame, drivers: pd.DataFrame,
                          driver_names: list[str],
                          start: str | None = None,
                          end: str | None = None) -> pd.DataFrame:
    """Long-format (date, driverName, DriverEloRating) for the given drivers."""
    df = _with_driver_names(results, drivers)
    df = df[df["driverName"].isin(driver_names)].copy()
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    df = df.sort_values("date")
    return df[["date", "driverName", "DriverEloRating", "name"]].rename(
        columns={"name": "race"}
    )


def constructor_rating_history(results: pd.DataFrame, constructors: pd.DataFrame,
                               constructor_names: list[str],
                               start: str | None = None,
                               end: str | None = None) -> pd.DataFrame:
    """Long-format (date, constructor, EloRating) for the given constructors."""
    df = _with_constructor_names(results, constructors)
    df = df[df["name"].isin(constructor_names)].copy()
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    df = df.sort_values("date")
    return df[["date", "name", "EloRating"]].rename(columns={"name": "constructor"})


# ---------------------------------------------------------------------------
# Distribution stats
# ---------------------------------------------------------------------------
def percentile_rating(values, percentile: float) -> float:
    """Fit a normal distribution to the ratings and return the Nth percentile."""
    mu, std = norm.fit(list(values))
    return float(norm.ppf(percentile / 100.0, loc=mu, scale=std))
