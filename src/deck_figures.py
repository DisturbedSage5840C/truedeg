"""Narrative figures for the idea-submission document.

    python -m src.deck_figures

Each function writes exactly one figure into figures/. These are the pictures
that carry the argument: raw data is messy -> tyre age is confounded -> control
for the confounders -> a clean degradation curve falls out -> it validates on
races the model never saw.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import load_features
from src.regression import (
    fit_pace_model, choose_fuel_mode, slope_progression, inchident_curve,
    _model_frame,
)

OUT = "figures"
RAW_RACE = "2023_Bahrain"      # high-deg, hard-tyre, long stints
VAL_RACES = [("2023_Bahrain", "HARD"), ("2023_Spain", "MEDIUM")]


def _pick_stint(df, race, compound, min_len=14):
    """Longest fresh-tyre stint of `compound` in `race`, tie-break on smoothness."""
    sub = df[(df.RaceId == race) & (df.Compound == compound) & df.FreshStint
             & (df.StintLen >= min_len)]
    if sub.empty:
        sub = df[(df.RaceId == race) & (df.Compound == compound) & df.FreshStint]
    scored = []
    for sid, g in sub.groupby("StintId"):
        g = g.sort_values("TyreLife")
        fit = np.polyval(np.polyfit(g.TyreLife, g.LapTimeS, 2), g.TyreLife)
        scored.append((np.std(g.LapTimeS - fit), len(g), sid))
    # cleanest first (residual std), break ties toward longer stints
    clean = [s for s in scored if s[0] <= 0.22] or scored
    clean.sort(key=lambda s: (s[0] > 0.22, -s[1]))
    return df[df.StintId == clean[0][2]].sort_values("TyreLife")


# ---------- SLIDE 3: raw pace per lap, one race ----------------------------
def raw_pace(df):
    race = df[df.RaceId == RAW_RACE]
    cand = race[race.FreshStint & (race.StintLen >= 18)]
    # 3 cleanest long stints (smallest scatter around a quadratic = not a backmarker)
    score = {}
    for sid, g in cand.groupby("StintId"):
        g = g.sort_values("TyreLife")
        fit = np.polyval(np.polyfit(g.TyreLife, g.LapTimeS, 2), g.TyreLife)
        score[sid] = np.std(g.LapTimeS - fit)
    stints = sorted(score, key=score.get)[:3]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for sid in stints:
        g = race[race.StintId == sid].sort_values("TyreLife")
        g = g[g.TyreLife <= g.TyreLife.iloc[0] + 25]
        pace = g.LapTimeS - g.LapTimeS.iloc[:3].mean()          # re-centre on start
        age = g.TyreLife - g.TyreLife.iloc[0]
        ax.plot(age, pace, marker="o", ms=4,
                label=f"{g.Driver.iloc[0]} ({g.Compound.iloc[0].lower()})")

    x = np.array([0, 25])
    ax.plot(x, 0.11 * x, "k--", lw=1.5,
            label="expected if only tyre age mattered (+0.11 s/lap)")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("Tyre age (laps into the stint)")
    ax.set_ylabel("Lap time vs start of stint (s)   -- lower is faster")
    ax.set_title(f"{RAW_RACE.replace('_', ' ')}: pace through single stints\n"
                 "tyres age 25 laps, lap time barely moves -- not the naive "
                 "degradation ramp")
    ax.legend(fontsize=8.5, title="driver (compound)", loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/deck_raw_pace.png", dpi=160)
    plt.close(fig)


# ---------- SLIDE 4-5: the collinearity / identifiability argument ---------
def collinearity(df):
    d = _model_frame(df)
    within = np.array([g[["TyreLife", "FuelKg"]].corr().iloc[0, 1]
                       for _, g in d.groupby("StintId") if len(g) >= 6],
                      dtype=float)
    within = within[np.isfinite(within)]
    within_med = np.median(within)
    pooled = d[["TyreLife", "FuelKg"]].corr().iloc[0, 1]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.6))

    # a few individual stints -- each is a perfectly straight line
    example = (d[d.StintId.isin(d.StintId.drop_duplicates().sample(
        6, random_state=3))])
    for _, g in example.groupby("StintId"):
        a0.plot(g.TyreLife, g.FuelKg, marker="o", ms=3, lw=1)
    a0.set_title(f"One stint at a time:  corr = {within_med:.2f}\n"
                 "tyre age and fuel are the same clock")
    a0.set_xlabel("Tyre age (laps)")
    a0.set_ylabel("Fuel load (kg, proxy)")

    samp = d.sample(min(5000, len(d)), random_state=0)
    a1.scatter(samp.TyreLife, samp.FuelKg, s=6, alpha=.25, color="#37c")
    a1.axvline(15, color="k", lw=1, ls=":")
    a1.set_title(f"13 races pooled:  corr = {pooled:.2f}\n"
                 "one tyre age now spans every fuel load")
    a1.set_xlabel("Tyre age (laps)")
    a1.set_ylabel("Fuel load (kg, proxy)")
    fig.suptitle("Why the method needs many races: pooling breaks the "
                 "tyre-age / fuel collinearity", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/deck_collinearity.png", dpi=160)
    plt.close(fig)


# ---------- SLIDE 8a: the tyre-age slope as controls are added ------------
def slope_bars(df):
    prog = slope_progression(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#c33" if s < 0 else "#3a7" for s in prog.slope_s_per_lap]
    ax.barh(range(len(prog)), prog.slope_s_per_lap, color=colors)
    ax.set_yticks(range(len(prog)))
    ax.set_yticklabels(prog.controls)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Estimated tyre-age effect on lap time (s/lap)")
    ax.set_title("Naive: tyres 'get faster' with age (wrong sign).\n"
                 "Add fuel + track evolution and the real +0.11 s/lap "
                 "degradation appears.")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/deck_slope_progression.png", dpi=160)
    plt.close(fig)


# ---------- SLIDE 8b: naive curve vs context-aware curve -----------------
def naive_vs_context(df):
    d = _model_frame(df)
    res, imposed, _ = choose_fuel_mode(df)

    # naive: raw pace vs race median, binned by tyre age (what you see directly)
    naive = d.assign(P=d.LapTimeS - d.groupby("RaceId").LapTimeS.transform("median"))
    nb = naive.groupby("TyreLife").P
    nb_med = nb.median()[nb.size() >= 20]
    nb_med = nb_med[nb_med.index <= 25] - nb_med.iloc[:3].mean()

    # context: counterfactual sweep on a representative mid-race context
    stint = _pick_stint(df, "2023_Spain", "MEDIUM")
    ctx = stint.iloc[0][["Event", "Compound", "FuelKg", "TrackEvoIdx",
                         "TrackTemp"]].to_dict()
    ages, deg = inchident_curve(res, ctx, max_age=25)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    a0.plot(nb_med.index, nb_med.values, marker="o", color="#888")
    a0.axhline(0, color="k", lw=.7)
    a0.set_title("Naive: raw pace vs tyre age (pooled)\n"
                 "fuel burn cancels degradation -- curve stays flat")
    a0.set_xlabel("Tyre age (laps)")
    a0.set_ylabel("Pace loss vs fresh tyre (s)")

    a1.plot(ages, deg, marker="o", color="#c33", lw=2)
    a1.axhline(0, color="k", lw=.7)
    a1.set_title("Context-aware: predicted tyre penalty\n"
                 "(fuel, evolution, temp held fixed -- only tyre age varies)")
    a1.set_xlabel("Tyre age (laps)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/deck_naive_vs_context.png", dpi=160)
    plt.close(fig)


# ---------- SLIDE 7: decompose one observed stint -----------------------
def _component(res, base_ctx, var, values):
    """Change in predicted pace as `var` sweeps `values`, all else held at ctx."""
    grid = pd.DataFrame([{**base_ctx, var: float(v)} for v in values])
    pred = res.predict(grid).to_numpy()
    return pred - pred[0]


def decomposition(df):
    res, imposed, _ = choose_fuel_mode(df)
    stint = _pick_stint(df, "2023_Bahrain", "HARD")
    g = stint.copy()
    a = g.TyreLife.to_numpy()
    ctx = g.iloc[0][["Event", "Compound", "FuelKg", "TrackEvoIdx",
                     "TrackTemp", "TyreLife"]].to_dict()

    tyre = _component(res, ctx, "TyreLife", a)
    fuel = _component(res, ctx, "FuelKg", g.FuelKg.to_numpy())
    evo = _component(res, ctx, "TrackEvoIdx", g.TrackEvoIdx.to_numpy())
    obs = (g.LapTimeS - g.LapTimeS.iloc[:3].mean()).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(a, obs, "k-o", ms=3, lw=1.5, label="observed lap time (vs stint start)")
    ax.plot(a, tyre, color="#c33", lw=2, label="tyre degradation (model)")
    ax.plot(a, fuel, color="#37c", lw=2, label="fuel burn (model)")
    ax.plot(a, evo + tyre + fuel, color="#3a7", lw=1.5, ls="--",
            label="model total")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap-time change vs start of stint (s)")
    ax.set_title(f"One stint decomposed -- {stint.Driver.iloc[0]}, "
                 f"{RAW_RACE.replace('_', ' ')} (hard)\n"
                 "fuel makes the car faster while the tyre makes it slower")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/deck_decomposition.png", dpi=160)
    plt.close(fig)


# ---------- SLIDE 9: validate on races the model never saw --------------
def _race_median_curve(df, race, comp, min_stints=4):
    """Fuel-corrected pace loss vs tyre age, median over every fresh stint of
    `comp` in `race`, each stint re-centred on its own first 3 laps."""
    sub = df[(df.RaceId == race) & (df.Compound == comp) & df.FreshStint
             & (df.StintLen >= 12)]
    parts = []
    for _, g in sub.groupby("StintId"):
        g = g.sort_values("TyreLife").copy()
        g["fcn"] = g.FuelCorrectedS - g.FuelCorrectedS.head(3).mean()
        parts.append(g)
    t = pd.concat(parts, ignore_index=True)
    grp = t.groupby("TyreLife")["fcn"]
    med = grp.median()[grp.size() >= min_stints]
    return med


def validation(df):
    fig, axes = plt.subplots(1, len(VAL_RACES), figsize=(11, 4.3), sharey=True)
    for ax, (race, comp) in zip(axes, VAL_RACES):
        train = df[df.RaceId != race]
        res, imposed, _ = choose_fuel_mode(train)

        # representative context for this race+compound (median over its stints)
        here = df[(df.RaceId == race) & (df.Compound == comp) & df.FreshStint]
        ctx = {"Event": here.Event.iloc[0], "Compound": comp,
               "FuelKg": here.FuelKg.median(), "TrackEvoIdx": here.TrackEvoIdx.median(),
               "TrackTemp": here.TrackTemp.median()}

        med = _race_median_curve(df, race, comp)
        ages, deg = inchident_curve(res, ctx, max_age=int(med.index.max()))

        ax.plot(med.index, med.values, marker="o", ms=4, color="#37c",
                label="actual (fuel-corrected, median of stints)")
        ax.plot(ages, deg, color="#c33", lw=2.5,
                label="predicted Inchident (race held out)")
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"{race.replace('_', ' ')} -- {comp.lower()}\n"
                     f"{here.StintId.nunique()} stints, model never saw this race")
        ax.set_xlabel("Tyre age (laps)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Pace loss vs fresh tyre (s)")
    fig.suptitle("Held-out check: the recovered curve has the right shape on "
                 "both races;\nthe single global rate runs high on the "
                 "lower-deg circuit (per-circuit slopes = next step)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/deck_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load_features()
    for fn in [raw_pace, collinearity, slope_bars, naive_vs_context,
               decomposition, validation]:
        print("running", fn.__name__)
        fn(data)
    print("deck figures written to", OUT)
