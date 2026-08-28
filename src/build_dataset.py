"""Build the TrueDeg stint dataset from FastF1.

This is the ONLY script that touches the network. Run once, cache forever.

    python -m src.build_dataset --smoke   # 2 races, sanity check first
    python -m src.build_dataset           # full RACES list

Never delete data/cache/ -- the first run is a long download, after that it is
near-instant.
"""
import argparse
import os
import warnings

import pandas as pd
import fastf1

warnings.filterwarnings("ignore")

CACHE = "data/cache"
os.makedirs(CACHE, exist_ok=True)
fastf1.Cache.enable_cache(CACHE)

# Deliberately contrasting circuits -- abrasion, layout, thermal environment.
RACES = [
    # (year, event, session)
    (2023, "Bahrain",     "R"),   # abrasive, hot, high-deg
    (2024, "Bahrain",     "R"),
    (2023, "Spain",       "R"),   # high-energy corners, high-deg
    (2024, "Spain",       "R"),
    (2023, "Silverstone", "R"),   # fast, cool, permanent
    (2024, "Silverstone", "R"),
    (2023, "Monza",       "R"),   # low-deg, low-downforce
    (2024, "Monza",       "R"),
    (2023, "Suzuka",      "R"),
    (2024, "Suzuka",      "R"),
    # --- held out for generalization testing ---
    (2024, "Jeddah",      "R"),   # smooth street
    (2024, "Monaco",      "R"),   # lowest-energy
    (2024, "Las Vegas",   "R"),   # cold night
]

# Small subset for the sanity-check pass described in the plan (step 1).
RACES_SMOKE = [
    (2023, "Bahrain", "R"),
    (2023, "Monza",   "R"),
]

WEATHER_COLS = ["AirTemp", "Humidity", "Pressure", "Rainfall",
                "TrackTemp", "WindDirection", "WindSpeed"]


def load_session(year, event, ses):
    s = fastf1.get_session(year, event, ses)
    # telemetry=False keeps this fast; turn on later for Experiment 7.
    s.load(laps=True, telemetry=False, weather=True, messages=False)
    return s


def extract(year, event, ses):
    s = load_session(year, event, ses)
    laps = s.laps
    if laps is None or laps.empty:
        return None
    laps = laps.reset_index(drop=True)

    # --- weather aligned per lap (FastF1 matches on timestamp) ---
    # Assign column-by-column with .values so `laps` stays a fastf1 Laps object
    # and keeps the .pick_* helpers used in clean().
    w = laps.get_weather_data().reset_index(drop=True)
    for col in WEATHER_COLS:
        laps[col] = w[col].values if col in w.columns else pd.NA

    # --- identity ---
    laps["Year"] = year
    laps["Event"] = event
    laps["Session"] = ses
    laps["RaceId"] = f"{year}_{event}"
    laps["StintId"] = (laps["RaceId"] + "_" + laps["Driver"].astype(str)
                       + "_S" + laps["Stint"].astype("Int64").astype(str))

    # --- numeric lap time ---
    laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()

    # --- session context needed for the fuel proxy ---
    total = getattr(s, "total_laps", None)
    laps["TotalLaps"] = total if total else int(laps["LapNumber"].max())
    return laps


def clean(laps):
    """Hard-removal rules. Prints before/after so you can show the effect."""
    n0 = len(laps)
    out = laps

    # FastF1's own lap-validity chain. Guard each step so an API rename in a
    # future FastF1 version degrades gracefully rather than killing the run.
    for step in ("pick_wo_box", "pick_accurate"):
        fn = getattr(out, step, None)
        if callable(fn):
            out = fn()
    ts = getattr(out, "pick_track_status", None)
    if callable(ts):
        try:
            out = ts("1", how="equals")          # green flag only
        except TypeError:
            out = out[out["TrackStatus"].astype(str) == "1"]
    else:
        out = out[out["TrackStatus"].astype(str) == "1"]

    out = pd.DataFrame(out)
    out = out[out["LapTimeS"].notna()]
    out = out[~out["Rainfall"].fillna(False).astype(bool)]   # dry running only
    # Drop red-flag / long-safety-car mega-stints: TyreLife carries across a
    # stoppage and F1 racing stints are ~5-35 laps. Keeps the deg study honest.
    out = out[out["TyreLife"] <= 40]
    print(f"    cleaned {n0} -> {len(out)} laps")
    return out


def iqr_filter(g):
    q1, q3 = g["LapTimeS"].quantile([0.25, 0.75])
    span = q3 - q1
    return g[(g["LapTimeS"] >= q1 - 1.5 * span) &
            (g["LapTimeS"] <= q3 + 1.5 * span)]


def main(races):
    frames = []
    for (y, e, ses) in races:
        try:
            print(f"[load] {y} {e} {ses}")
            raw = extract(y, e, ses)
            if raw is None:
                print("    !! no laps")
                continue
            frames.append(clean(raw))
        except Exception as err:
            print(f"    !! skipped {y} {e}: {err}")

    if not frames:
        raise SystemExit("no data extracted -- check network / FastF1 cache")

    df = pd.concat(frames, ignore_index=True)

    # per-stint IQR outlier removal on lap time
    df = pd.concat(
        [iqr_filter(g) for _, g in df.groupby("StintId", sort=False)],
        ignore_index=True,
    )

    # only keep stints long enough to show a trend
    keep = df.groupby("StintId")["LapNumber"].transform("size") >= 6
    df = df[keep].reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/stints.parquet")
    print(f"\nDONE  {len(df)} laps | {df.StintId.nunique()} stints "
          f"| {df.RaceId.nunique()} races")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="only build the 2-race sanity subset")
    args = ap.parse_args()
    main(RACES_SMOKE if args.smoke else RACES)
