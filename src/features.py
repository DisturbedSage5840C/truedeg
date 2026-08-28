"""Derived features / proxies for the TrueDeg experiments and models.

Everything downstream (experiments.py, model.py, app.py) reads
data/stints.parquet through add_features().
"""
import numpy as np
import pandas as pd

FUEL_START_KG = 110.0
FUEL_S_PER_KG = 0.03      # ~0.3 s per 10 kg -- a PRIOR, not a measurement.


def add_features(df):
    df = df.copy()

    # --- fuel proxy: linear burn from race start to flag ---
    frac = (df["LapNumber"] / df["TotalLaps"]).clip(0, 1)
    df["FuelKg"] = FUEL_START_KG * (1 - frac)
    df["FuelEffectS"] = FUEL_S_PER_KG * df["FuelKg"]

    # --- track evolution: cumulative field laps completed before this lap ---
    sort_key = "LapStartTime" if "LapStartTime" in df.columns else "LapNumber"
    df = df.sort_values(["RaceId", sort_key]).reset_index(drop=True)
    df["FieldLapsSoFar"] = df.groupby("RaceId").cumcount()
    df["TrackEvoIdx"] = df.groupby("RaceId")["FieldLapsSoFar"].transform(
        lambda x: x / x.max() if x.max() else 0.0)

    # --- thermal ---
    df["TrackAirDelta"] = df["TrackTemp"] - df["AirTemp"]

    # --- stint shape: did it start on a fresh tyre, how deep are we in it ---
    df["StintStartTyreLife"] = df.groupby("StintId")["TyreLife"].transform("min")
    df["FreshStint"] = df["StintStartTyreLife"] <= 3     # lap 1 is an out-lap
    df["StintLap"] = df["TyreLife"] - df["StintStartTyreLife"]
    df["StintLen"] = df.groupby("StintId")["LapNumber"].transform("size")

    # --- per-stint reference pace (fastest lap of the stint) ---
    df["StintBestS"] = df.groupby("StintId")["LapTimeS"].transform("min")
    df["DeltaToStintBest"] = df["LapTimeS"] - df["StintBestS"]

    # --- per-race reference, to compare drivers/teams fairly ---
    df["RaceMedianS"] = df.groupby("RaceId")["LapTimeS"].transform("median")
    df["PaceVsRaceMedian"] = df["LapTimeS"] - df["RaceMedianS"]

    # --- naive fuel-corrected pace (used in Experiment 1) ---
    df["FuelCorrectedS"] = df["LapTimeS"] - df["FuelEffectS"]

    return df


def load_features(path="data/stints.parquet"):
    return add_features(pd.read_parquet(path))
