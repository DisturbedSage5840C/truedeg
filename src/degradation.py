"""Degradation coefficients -- the quantity TrueDeg actually estimates.

    python -m src.degradation

Lap time is the instrument, not the target. We write observed lap time as

    LapTime = circuit baseline
            + d(circuit, compound, track temp) * tyre_age   # <- the unknown
            + curvature * tyre_age^2
            + fuel effect + track evolution + thermal + noise

and invert it for the degradation coefficient d. This module fits the
per-circuit and temperature-modulated versions of d and writes the two figures
the idea-submission document is built around.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from src.features import load_features

OUT = "figures"

# circuits present in both 2023 and 2024 -- their coefficient is cross-checkable
PAIRED = {"Bahrain", "Spain", "Silverstone", "Monza", "Suzuka"}

CIRCUIT_NOTE = {
    "Suzuka":      "fast, high-energy corners",
    "Monza":       "low downforce, heavy braking",
    "Bahrain":     "abrasive surface, hot",
    "Las Vegas":   "cold night, long straights",
    "Silverstone": "fast, cool, flowing",
    "Spain":       "sustained-load corners",
    "Jeddah":      "smooth street, low abrasion",
    "Monaco":      "lowest energy, tyre-management race",
}

PER_CIRCUIT_FORMULA = (
    "LapTimeS ~ C(Event) + C(Event):TyreLife + I(TyreLife**2)"
    " + FuelKg + TrackEvoIdx + C(Compound)"
)
TEMP_FORMULA = (
    "LapTimeS ~ C(Event) + TyreLife + I(TyreLife**2) + TyreLife:TrackTempC"
    " + FuelKg + TrackEvoIdx + TrackTemp + C(Compound)"
)


def _frame(df):
    cols = ["LapTimeS", "Event", "TyreLife", "FuelKg", "TrackEvoIdx",
            "TrackTemp", "AirTemp", "Compound", "RaceId"]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d.TyreLife <= 40].copy()
    d["TrackTempC"] = d.TrackTemp - d.TrackTemp.mean()
    return d


def per_circuit_coefficients(df):
    """Linear tyre-age coefficient d per circuit, full context held around it."""
    d = _frame(df)
    res = smf.ols(PER_CIRCUIT_FORMULA, data=d).fit()
    ci = res.conf_int()
    rows = []
    for term in res.params.index:
        if ":TyreLife" not in term or "I(" in term:
            continue
        circuit = re.search(r"\[(?:T\.)?([^\]]+)\]", term).group(1)
        rows.append({
            "circuit": circuit,
            "d_s_per_lap": res.params[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "paired": circuit in PAIRED,
        })
    out = pd.DataFrame(rows).sort_values("d_s_per_lap", ascending=False)
    return out.reset_index(drop=True), res


def temperature_sensitivity(df):
    """How the degradation coefficient moves with track temperature."""
    d = _frame(df)
    res = smf.ols(TEMP_FORMULA, data=d).fit()
    base = res.params["TyreLife"]
    per_degc = res.params["TyreLife:TrackTempC"]
    ci = res.conf_int().loc["TyreLife:TrackTempC"]
    tmean = df["TrackTemp"].replace([np.inf, -np.inf], np.nan).dropna()
    tmean = tmean[df["TyreLife"] <= 40].mean()
    return {
        "d_at_mean_temp": base,
        "d_per_degC": per_degc,
        "ci": (ci[0], ci[1]),
        "p": res.pvalues["TyreLife:TrackTempC"],
        "mean_track_temp": tmean,
        "curvature": res.params.get("I(TyreLife ** 2)", 0.0),
    }, res


def fig_by_circuit(tbl):
    hi, mid = 0.095, 0.06
    colour = ["#c63f27" if v >= hi else "#d98b3a" if v >= mid else "#2f6f7c"
              for v in tbl.d_s_per_lap]
    y = np.arange(len(tbl))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh(y, tbl.d_s_per_lap, color=colour, height=.62)
    ax.errorbar(tbl.d_s_per_lap, y,
                xerr=[tbl.d_s_per_lap - tbl.ci_low, tbl.ci_high - tbl.d_s_per_lap],
                fmt="none", ecolor="#333", elinewidth=1, capsize=3)
    for yi, (_, r) in zip(y, tbl.iterrows()):
        tag = CIRCUIT_NOTE.get(r.circuit, "")
        mark = "" if r.paired else "  (one season)"
        ax.text(r.ci_high + 0.004, yi,
                f"{r.d_s_per_lap:.3f} s/lap   {tag}{mark}",
                va="center", fontsize=8.5, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels(tbl.circuit)
    ax.set_xlim(0, tbl.ci_high.max() + 0.075)
    ax.set_xlabel("estimated tyre-degradation coefficient  d  (seconds lost per lap of tyre age)")
    ax.set_title("Degradation coefficient by circuit -- context held fixed, only tyre age swept")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/deg_by_circuit.png", dpi=160)
    plt.close(fig)


def fig_temp_sensitivity(info):
    d0 = info["d_at_mean_temp"]
    slope = info["d_per_degC"]
    quad = info["curvature"]
    ages = np.arange(31)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for dt, label, c in [(-12, "cool track  (~-12 degC)", "#2f6f7c"),
                         (0,  "field-average track", "#555"),
                         (+12, "hot track  (~+12 degC)", "#c63f27")]:
        d = d0 + slope * dt
        pen = d * ages + quad * ages ** 2
        ax.plot(ages, pen, color=c, lw=2, label=label)
    ax.set_xlabel("tyre age (laps)")
    ax.set_ylabel("estimated tyre-degradation penalty (s)")
    ax.set_title("Same method, temperature-modulated coefficient\n"
                 f"d moves {slope:+.4f} s/lap per degC of track temp "
                 f"(p = {info['p']:.0e})")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/deg_temp_sensitivity.png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_features()

    tbl, _ = per_circuit_coefficients(df)
    print("=== degradation coefficient by circuit (s/lap) ===")
    for r in tbl.itertuples():
        flag = "" if r.paired else "  [one season -- not cross-validated]"
        print(f"  {r.circuit:14s} {r.d_s_per_lap:+.4f}  "
              f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]{flag}")

    info, _ = temperature_sensitivity(df)
    print("\n=== temperature sensitivity of the coefficient ===")
    print(f"  d at field-average track temp : {info['d_at_mean_temp']:+.4f} s/lap")
    print(f"  change per degC of track temp : {info['d_per_degC']:+.5f} s/lap  "
          f"(95% CI [{info['ci'][0]:+.5f}, {info['ci'][1]:+.5f}], p={info['p']:.1e})")
    print(f"  mean track temp in the data   : {info['mean_track_temp']:.1f} degC")

    fig_by_circuit(tbl)
    fig_temp_sensitivity(info)
    print(f"\nwrote {OUT}/deg_by_circuit.png, {OUT}/deg_temp_sensitivity.png")


if __name__ == "__main__":
    main()
