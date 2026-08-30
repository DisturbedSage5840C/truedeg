"""Interpretable OLS pace model -- the headline evidence for the submission.

    python -m src.regression

Everything here is a plain linear regression you can read coefficient by
coefficient. The XGBoost model in src/model.py is the "and a flexible model
does even better" note; this is the "and every number is inspectable" core.

The identifiability argument
---------------------------
Within one stint, tyre age and fuel load are both linear in lap number, so
they are ~collinear (corr about -0.99) and no single stint can tell tyre
degradation apart from fuel-burn gain. Pooling stints that start at different
race laps -- a lap-2 stint and a lap-34 stint sit at very different fuel loads
at the same tyre age -- breaks that collinearity and lets OLS attribute the
variance. That is why the method needs many races, not one.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from src.features import load_features, FUEL_S_PER_KG

# Circuit fixed effects (C(Event)) absorb the ~20 s between-track spread so the
# physics terms below are estimated purely from within-race variation.
FULL_FORMULA = (
    "LapTimeS ~ C(Event) + TyreLife + I(TyreLife**2) + FuelKg"
    " + TrackEvoIdx + TrackTemp + C(Compound)"
)
NAIVE_FORMULA = "LapTimeS ~ C(Event) + TyreLife + C(Compound)"

# When impose_fuel=True we move the fuel term to a fixed offset (the 0.03 s/kg
# prior from features.py) instead of estimating it.
IMPOSED_FULL_FORMULA = (
    "PaceMinusFuel ~ C(Event) + TyreLife + I(TyreLife**2)"
    " + TrackEvoIdx + TrackTemp + C(Compound)"
)

PHYS_TERMS = ["TyreLife", "I(TyreLife ** 2)", "FuelKg", "TrackEvoIdx", "TrackTemp"]


def _model_frame(df):
    """Rows the regression is allowed to see: finite in every RHS column."""
    cols = ["LapTimeS", "Event", "TyreLife", "FuelKg", "TrackEvoIdx",
            "TrackTemp", "Compound", "RaceId", "StintId", "LapNumber",
            "FuelEffectS", "FuelCorrectedS", "PaceVsRaceMedian"]
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["LapTimeS", "TyreLife", "FuelKg", "TrackEvoIdx", "TrackTemp"])
    d["PaceMinusFuel"] = d["LapTimeS"] - d["FuelEffectS"]
    return d


def fit_pace_model(df, impose_fuel=False):
    d = _model_frame(df)
    formula = IMPOSED_FULL_FORMULA if impose_fuel else FULL_FORMULA
    return smf.ols(formula, data=d).fit()


def fit_naive_model(df):
    return smf.ols(NAIVE_FORMULA, data=_model_frame(df)).fit()


def coefficient_table(res):
    """Physics coefficients with 95% CIs -- the inspectable core."""
    ci = res.conf_int()
    rows = []
    for term in res.params.index:
        if term.startswith("C(Event)") or term == "Intercept":
            continue
        rows.append({
            "term": term,
            "estimate": res.params[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "p": res.pvalues[term],
        })
    return pd.DataFrame(rows)


def fuel_prior_check(res):
    """Estimated fuel effect (s per kg) vs the independent 0.03 s/kg prior."""
    if "FuelKg" not in res.params.index:
        return {"estimated_s_per_kg": None, "prior_s_per_kg": FUEL_S_PER_KG,
                "note": "fuel imposed as a fixed offset, not estimated"}
    est = res.params["FuelKg"]
    ci = res.conf_int().loc["FuelKg"]
    return {
        "estimated_s_per_kg": est,
        "ci": (ci[0], ci[1]),
        "prior_s_per_kg": FUEL_S_PER_KG,
        "ratio_to_prior": est / FUEL_S_PER_KG,
        # a lighter car is faster, so LapTime should RISE with FuelKg -> est > 0
        "sign_ok": est > 0,
    }


def _predict_pace(res, ctx, ages):
    """Predicted LapTime (or PaceMinusFuel) across a tyre-age sweep."""
    grid = pd.DataFrame([{**ctx, "TyreLife": float(a)} for a in ages])
    return res.predict(grid).to_numpy()


def inchident_curve(res, context_row, max_age=25):
    """Counterfactual sweep: hold context fixed, vary only TyreLife.

    penalty(a) = predicted_pace(a) - predicted_pace(0)
    """
    ages = np.arange(max_age + 1)
    pred = _predict_pace(res, dict(context_row), ages)
    return ages, pred - pred[0]


def groupkfold_mae(df, fit_fn):
    """Leave-one-race-out MAE, scored on LapTimeS.

    Restricted to circuits with two seasons in the data, so a held-out race's
    circuit fixed effect is still estimated from the other season -- otherwise
    C(Event) has no level for it and the score is meaningless.
    """
    d = _model_frame(df)
    paired = d["Event"].value_counts()
    paired = [e for e in d["Event"].unique()
              if d.loc[d.Event == e, "RaceId"].nunique() >= 2]
    d = d[d["Event"].isin(paired)]
    errs = []
    for rid in d["RaceId"].unique():
        te = d["RaceId"] == rid
        res = fit_fn(d[~te])
        pred = res.predict(d[te])
        errs.append(np.mean(np.abs(d.loc[te, "LapTimeS"] - pred)))
    return float(np.mean(errs)), float(np.std(errs))


def slope_progression(df):
    """Tyre-age coefficient (s/lap) as physical controls are added one by one.

    On raw F1 data the naive slope is the WRONG SIGN -- older tyres look faster,
    because tyre age is confounded with fuel burn and track evolution. The true
    degradation only appears once those are in the model.
    """
    d = _model_frame(df)
    steps = [
        ("tyre age only", "LapTimeS ~ TyreLife"),
        ("+ circuit", "LapTimeS ~ C(Event) + TyreLife"),
        ("+ compound", "LapTimeS ~ C(Event) + C(Compound) + TyreLife"),
        ("+ track temp", "LapTimeS ~ C(Event) + C(Compound) + TrackTemp + TyreLife"),
        ("+ track evolution",
         "LapTimeS ~ C(Event) + C(Compound) + TrackTemp + TrackEvoIdx + TyreLife"),
        ("+ fuel  (full model)",
         "LapTimeS ~ C(Event) + C(Compound) + TrackTemp + TrackEvoIdx + FuelKg"
         " + TyreLife + I(TyreLife**2)"),
    ]
    out = []
    for label, formula in steps:
        r = smf.ols(formula, data=d).fit()
        ci = r.conf_int().loc["TyreLife"]
        out.append({"controls": label, "slope_s_per_lap": r.params["TyreLife"],
                    "ci_low": ci[0], "ci_high": ci[1]})
    return pd.DataFrame(out)


def choose_fuel_mode(df):
    """Estimate fuel if the data identifies it sensibly, else impose the prior."""
    res = fit_pace_model(df, impose_fuel=False)
    chk = fuel_prior_check(res)
    ok = chk["sign_ok"] and 0.25 <= chk["ratio_to_prior"] <= 4.0
    if ok:
        return res, False, chk
    return fit_pace_model(df, impose_fuel=True), True, chk


def main():
    os.makedirs("figures", exist_ok=True)
    df = load_features()

    res, imposed, chk = choose_fuel_mode(df)
    mode = "IMPOSED 0.03 s/kg prior" if imposed else "ESTIMATED from data"

    print(f"\n=== OLS pace model  (fuel: {mode}) ===")
    print(f"    n = {int(res.nobs)} laps   adj R^2 = {res.rsquared_adj:.3f}")
    print("\nphysics coefficients (95% CI):")
    tbl = coefficient_table(res)
    for r in tbl.itertuples():
        print(f"  {r.term:20s} {r.estimate:+8.4f}  "
              f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]  p={r.p:.1e}")

    print("\nfuel-prior check:")
    for k, v in chk.items():
        print(f"  {k}: {v}")

    lin = res.params["TyreLife"]
    quad = res.params.get("I(TyreLife ** 2)", 0.0)
    print(f"\ntyre degradation: {lin:+.4f} s/lap linear, {quad:+.5f} s/lap^2 curvature")
    print(f"  -> modelled loss at 20 laps = {lin*20 + quad*400:+.2f} s")

    print("\ntyre-age slope as controls are added (the naive model is wrong-signed):")
    prog = slope_progression(df)
    for r in prog.itertuples():
        print(f"  {r.controls:24s} {r.slope_s_per_lap:+.4f} s/lap")

    m_full = groupkfold_mae(df, lambda d: fit_pace_model(d, impose_fuel=imposed))
    m_naive = groupkfold_mae(df, fit_naive_model)
    print(f"\nGroupKFold-by-race MAE (LapTimeS):")
    print(f"  naive  (age + compound + circuit) : {m_naive[0]:.3f} s  (+/- {m_naive[1]:.3f})")
    print(f"  context (+ fuel, evo, temp, curve): {m_full[0]:.3f} s  (+/- {m_full[1]:.3f})")
    print(f"  improvement: {100*(m_naive[0]-m_full[0])/m_naive[0]:.1f}%")

    # coefficient figure -- the terms with interpretable per-unit meaning and
    # tight CIs. TrackEvoIdx is a unitless index and is not separately identified
    # once fuel is in the model (see the printed table), so it is left off here.
    fig_terms = ["TyreLife", "I(TyreLife ** 2)", "FuelKg", "TrackTemp"]
    labels = {"TyreLife": "tyre age (s/lap)",
              "I(TyreLife ** 2)": "tyre age^2 (s/lap^2)",
              "FuelKg": "fuel (s/kg)", "TrackTemp": "track temp (s/degC)"}
    show = tbl.set_index("term").loc[fig_terms].reset_index()
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.errorbar(show.estimate, y,
                xerr=[show.estimate - show.ci_low, show.ci_high - show.estimate],
                fmt="o", capsize=4, color="#c33")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[t] for t in show.term])
    ax.set_xlabel("coefficient estimate  (effect on lap time, seconds)")
    ax.set_title(f"OLS pace-model coefficients -- fuel {mode.lower()}")
    fig.tight_layout()
    fig.savefig("figures/deck_coefficients.png", dpi=160)
    plt.close(fig)
    print("\nwrote figures/deck_coefficients.png")


if __name__ == "__main__":
    main()
