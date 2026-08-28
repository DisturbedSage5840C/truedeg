"""Experiments 0-6. Each function produces exactly ONE figure for the deck.

    python -m src.experiments
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import load_features

OUT = "figures"

# Circuits with two seasons of data. The other three (Jeddah, Monaco, Las Vegas)
# are single-season generalization hold-outs and too sparse / evolution-dominated
# for a clean per-circuit degradation curve.
CORE_EVENTS = ["Bahrain", "Spain", "Silverstone", "Monza", "Suzuka"]


def _recentre(df, cols, compound=None, events=None, min_len=12, max_age=28):
    """Fresh-tyre stints, each column re-centred on the stint's first 3 laps so
    every stint starts at ~0. Per-stint re-centring cancels circuit identity
    (a Monza lap and a Spain lap are ~20 s apart in absolute time), so stints
    from different races pool onto one comparable axis. Returns one tidy frame."""
    sub = df[df.FreshStint & (df.StintLen >= min_len) & (df.TyreLife <= max_age)]
    if compound is not None:
        sub = sub[sub.Compound == compound]
    if events is not None:
        sub = sub[sub.Event.isin(events)]
    parts = []
    for sid, g in sub.copy().groupby("StintId"):
        g = g.sort_values("TyreLife").copy()
        for c in cols:
            g[c + "_n"] = g[c] - g[c].head(3).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else sub.copy()


def _median_curve(tidy, col, min_stints=5):
    """Median of the re-centred column by tyre age, dropping thin tails."""
    grp = tidy.groupby("TyreLife")[col + "_n"]
    med = grp.median()
    return med[grp.size() >= min_stints]


# ---------- HYPOTHESIS 0: does raw data look like the clean textbook curve? ----
def exp0_hypothesis(df):
    tidy = _recentre(df, ["LapTimeS"], compound="MEDIUM")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    a = np.arange(0, 26)
    ax[0].plot(a, 0.03 * a, lw=2)
    ax[0].set_title("What we assume")
    ax[0].set_xlabel("Tyre age (laps)")
    ax[0].set_ylabel("Pace loss (s)")
    ax[0].set_ylim(-2.5, 3.0)

    for _, g in tidy.groupby("StintId"):
        ax[1].plot(g["TyreLife"], g["LapTimeS_n"], alpha=0.25, lw=1)
    med = _median_curve(tidy, "LapTimeS")
    ax[1].plot(med.index, med.values, color="k", lw=2.5, label="median")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("What real F1 stints look like (fresh MEDIUM)")
    ax[1].set_xlabel("Tyre age (laps)")
    ax[1].set_ylabel("Raw pace vs start of stint (s)")
    ax[1].set_ylim(-2.5, 3.0)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(f"{OUT}/exp0_hypothesis_vs_reality.png", dpi=160)
    plt.close(fig)


# ---------- EXPERIMENT 1: fuel ------------------------------------------------
def exp1_fuel(df):
    """Headline figure. Same stints, before and after fuel correction."""
    tidy = _recentre(df, ["LapTimeS", "FuelCorrectedS"], compound="MEDIUM")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for _, g in tidy.groupby("StintId"):
        ax[0].plot(g["TyreLife"], g["LapTimeS_n"], alpha=.22, lw=1)
        ax[1].plot(g["TyreLife"], g["FuelCorrectedS_n"], alpha=.22, lw=1)

    raw = _median_curve(tidy, "LapTimeS")
    fc = _median_curve(tidy, "FuelCorrectedS")
    ax[0].plot(raw.index, raw.values, color="k", lw=2.5)
    ax[1].plot(fc.index, fc.values, color="k", lw=2.5)

    ax[0].set_title("Raw lap time -- fuel burn hides the degradation")
    ax[1].set_title("After fuel correction -- degradation appears")
    for a in ax:
        a.axhline(0, color="k", lw=.8)
        a.set_xlabel("Tyre age (laps)")
        a.set_ylim(-2.5, 3.0)
    ax[0].set_ylabel("Pace vs start of stint (s)   (black = median)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp1_fuel.png", dpi=160)
    plt.close(fig)


# ---------- EXPERIMENT 2: track evolution -----------------------------------
def exp2_track_evolution(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rid, g in df.groupby("RaceId"):
        b = g.groupby(pd.cut(g["TrackEvoIdx"], 12),
                      observed=False)["LapTimeS"].median()
        vals = b.values
        if len(vals) == 0 or np.isnan(vals[0]):
            continue
        ax.plot(np.arange(len(vals)), vals - vals[0], marker="o",
                label=rid, lw=1.5)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylim(-3.5, 1.2)   # clip late safety-car / VSC bunching tails
    ax.set_xlabel("Session progression (binned)")
    ax.set_ylabel("Median field pace vs start (s)")
    ax.set_title("Track evolution: the whole field gets faster, "
                 "independent of any one tyre")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_track_evolution.png", dpi=160)
    plt.close(fig)


# ---------- EXPERIMENT 3: thermal ------------------------------------------
def exp3_thermal(df):
    """Hold tyre age roughly constant, vary track temp."""
    band = df[df.TyreLife.between(8, 12)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(band["TrackTemp"], band["PaceVsRaceMedian"],
                    c=band["AirTemp"], s=8, alpha=.5, cmap="coolwarm")
    ax.set_xlabel("Track temperature (deg C)")
    ax.set_ylabel("Pace vs race median (s)")
    ax.set_title("Tyre age fixed at 8-12 laps -- pace still varies with temp")
    plt.colorbar(sc, label="Air temp (deg C)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp3_thermal.png", dpi=160)
    plt.close(fig)


# ---------- EXPERIMENT 4: compound ---------------------------------------
def exp4_compound(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # SOFT stints are short, so a lower length floor / age ceiling than MEDIUM.
    spec = {"SOFT": (8, 20), "MEDIUM": (12, 28), "HARD": (14, 32)}
    for comp, (min_len, max_age) in spec.items():
        tidy = _recentre(df, ["FuelCorrectedS"], compound=comp,
                         events=CORE_EVENTS, min_len=min_len, max_age=max_age)
        if tidy.empty:
            continue
        med = _median_curve(tidy, "FuelCorrectedS", min_stints=4)
        if len(med) < 6:
            continue
        ax.plot(med.index, med.values, marker="o", label=comp)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Fuel-corrected pace loss vs start of stint (s)")
    ax.set_title("Degradation shape depends on compound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp4_compound.png", dpi=160)
    plt.close(fig)


# ---------- EXPERIMENT 5: circuit ---------------------------------------
def exp5_circuit(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tidy = _recentre(df, ["FuelCorrectedS"], compound="MEDIUM",
                     events=CORE_EVENTS, max_age=25)
    for ev, g in tidy.groupby("Event"):
        med = _median_curve(g, "FuelCorrectedS", min_stints=4)
        if len(med) < 8:
            continue
        ax.plot(med.index, med.values, marker=".", label=ev)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Fuel-corrected pace loss vs start of stint (s)")
    ax.set_title("Same compound, same age -- different circuits, "
                 "different degradation")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp5_circuit.png", dpi=160)
    plt.close(fig)


# ---------- EXPERIMENT 6: traffic (proxy via track position) --------------
def exp6_traffic(df):
    """Without gap data loaded, Position is a usable first proxy for dirty air."""
    band = df[df.TyreLife.between(6, 14)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b = band.groupby(pd.cut(band["Position"], [0, 4, 8, 12, 20]),
                     observed=False)["PaceVsRaceMedian"].median()
    ax.bar([str(i) for i in b.index], b.values)
    ax.set_xlabel("Track position band")
    ax.set_ylabel("Median pace vs race median (s)")
    ax.set_title("Tyre age fixed -- pace still varies with position in the pack")
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp6_traffic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load_features()
    for fn in [exp0_hypothesis, exp1_fuel, exp2_track_evolution,
               exp3_thermal, exp4_compound, exp5_circuit, exp6_traffic]:
        print("running", fn.__name__)
        fn(data)
    print("figures written to", OUT)
