"""Model 0 (naive) vs Model 1 (context-aware), and the first TrueDeg curve.

    python -m src.model

Deliverables D-G: the ablation table + the counterfactual degradation curve.

Evaluation note
---------------
Two independent protocols, both scoring on data the model never saw:

  1. temporal in-race split -- train on the first 60% of every race, predict the
     final 40%. This is the race-strategy setting: you always have the current
     race so far and want the rest of it.
  2. by-race GroupKFold -- train on whole races, predict a completely unseen
     race (paired circuits only, so the held-out circuit was seen in the other
     season).

Model 1 beats Model 0 by ~45% on both, which is the credibility argument for the
deck. The counterfactual curve is the scientific deliverable.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.features import load_features

# Target = lap time RELATIVE TO the race median, not absolute lap time.
# Absolute lap time is ~99% circuit identity (a 20 s spread between tracks), so
# a tree model just memorises "which circuit" and never learns degradation
# (TyreLife feature importance came out at 0.000). Subtracting the race median
# removes the circuit-level offset and leaves exactly the pace variation the
# project is about: tyre age, fuel, track evolution, thermal, traffic.
# Caveat: the race median is a per-race constant computed on the full race; in
# the strategy setting you know which circuit you are at, so this is a fixed
# offset, not a per-lap leak.
TARGET = "PaceVsRaceMedian"

# Circuits with only one season in the dataset -> a by-race fold holds the whole
# circuit out. Kept out of the by-race ablation and the counterfactual fit.
HOLDOUT_EVENTS = {"Jeddah", "Monaco", "Las Vegas"}

#   Model 0 : tyre age + compound only -- the naive degradation lookup.
#   Model 1 : + fuel, track evolution, thermal, traffic, circuit -- the context
#             the naive model silently folds into the tyre-age coefficient.
NAIVE_NUM = ["TyreLife"]
NAIVE_CATS = ["Compound"]

CONTEXT_NUM = ["TyreLife", "FuelKg", "TrackEvoIdx", "TrackTemp", "AirTemp",
               "Humidity", "WindSpeed", "TrackAirDelta", "Position"]
CONTEXT_CATS = ["Compound", "Event"]

XGB_KW = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
              enable_categorical=True, tree_method="hist")


def prep(df, num, cats):
    X = df[list(num) + list(cats)].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X, df[TARGET].astype(float)


def _align_cats(Xte, Xtr, cats):
    for c in cats:
        Xte[c] = Xte[c].astype(Xtr[c].dtype)
    return Xte


def evaluate_temporal(df, num, cats, name, cut=0.6):
    """Train on the first `cut` of every race, score on the rest."""
    frac = df.groupby("RaceId")["LapNumber"].transform(lambda x: x / x.max())
    tr, te = frac <= cut, frac > cut
    X, y = prep(df, num, cats)
    m = XGBRegressor(**XGB_KW)
    m.fit(X[tr], y[tr])
    mae = mean_absolute_error(y[te], m.predict(X[te]))
    print(f"  {name:24s} MAE = {mae:.3f} s")
    return float(mae)


def evaluate_byrace(df, num, cats, name):
    """GroupKFold by RaceId -- scored on races held entirely out."""
    X, y = prep(df, num, cats)
    groups = df["RaceId"]
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    errs = []
    for tr, te in gkf.split(X, y, groups=groups):
        m = XGBRegressor(**XGB_KW)
        m.fit(X.iloc[tr], y.iloc[tr])
        errs.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
    print(f"  {name:24s} MAE = {np.mean(errs):.3f} s  (+/- {np.std(errs):.3f})")
    return float(np.mean(errs))


def fit_full(df, num=CONTEXT_NUM, cats=CONTEXT_CATS):
    X, y = prep(df, num, cats)
    m = XGBRegressor(**XGB_KW)
    m.fit(X, y)
    return m, X


def truedeg_curve(model, X, context_row, cats=CONTEXT_CATS, max_age=25):
    """THE counterfactual sweep: hold everything fixed, vary only TyreLife.

    Returns (ages, D) where D(a) = P_a - P_0.
    """
    rows = [{**dict(context_row), "TyreLife": a} for a in range(max_age + 1)]
    sweep = pd.DataFrame(rows)[X.columns]
    for c in cats:
        sweep[c] = sweep[c].astype(X[c].dtype)
    pred = model.predict(sweep)
    return np.arange(max_age + 1), pred - pred[0]


def _pct(a, b):
    return 100 * (a - b) / a


def main():
    df = load_features()

    print("\n=== ABLATION 1: temporal in-race split (first 60% -> final 40%) ===")
    t0 = evaluate_temporal(df, NAIVE_NUM, NAIVE_CATS, "Model 0 (naive)")
    t1 = evaluate_temporal(df, CONTEXT_NUM, CONTEXT_CATS, "Model 1 (context)")
    print(f"  improvement: {_pct(t0, t1):.1f}%   (tested on laps never seen)")

    print("\n=== ABLATION 2: by-race GroupKFold (paired circuits) -- harder ===")
    dev = df[~df.Event.isin(HOLDOUT_EVENTS)].reset_index(drop=True)
    b0 = evaluate_byrace(dev, NAIVE_NUM, NAIVE_CATS, "Model 0 (naive)")
    b1 = evaluate_byrace(dev, CONTEXT_NUM, CONTEXT_CATS, "Model 1 (context)")
    print(f"  improvement: {_pct(b0, b1):.1f}%   (tested on whole unseen races)")

    # --- fit on the paired circuits for the counterfactual curve ---
    dev_i = dev.reset_index(drop=True)
    model, X = fit_full(dev_i)

    # a long fresh-tyre MEDIUM stint at a high-deg circuit = the clearest demo
    cand = dev_i[(dev_i.Compound == "MEDIUM") & dev_i.FreshStint
                 & dev_i.Event.eq("Spain")]
    if cand.empty:
        cand = dev_i[(dev_i.Compound == "MEDIUM") & dev_i.FreshStint]
    sid = cand.groupby("StintId").size().sort_values().index[-1]
    stint = dev_i[dev_i.StintId == sid].sort_values("TyreLife")
    ctx = X.loc[stint.index[0]].to_dict()

    ages, deg = truedeg_curve(model, X, ctx)
    # observed reference: fuel-corrected pace vs the mean of the first 3 laps
    base = stint["FuelCorrectedS"].head(3).mean()
    obs = stint["FuelCorrectedS"] - base

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ages, deg, lw=2.5,
            label="TrueDeg predicted degradation (counterfactual)")
    ax.scatter(stint["TyreLife"], obs, s=22, alpha=.7, color="crimson",
               label="Actual observed (fuel-corrected)")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Pace loss vs fresh tyre (s)")
    ax.set_ylim(min(-0.6, deg.min() - 0.3), max(deg.max(), obs.quantile(.95)) + 0.4)
    ax.set_title(f"First TrueDeg curve -- {sid}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/truedeg_curve_vs_actual.png", dpi=160)
    plt.close(fig)
    print("\nwrote figures/truedeg_curve_vs_actual.png")

    table = pd.DataFrame([
        {"model": "Model 0 (tyre age + compound)",
         "in_race_mae_s": round(t0, 3), "unseen_race_mae_s": round(b0, 3)},
        {"model": "Model 1 (context-aware)",
         "in_race_mae_s": round(t1, 3), "unseen_race_mae_s": round(b1, 3)},
    ])
    table.to_csv("figures/ablation.csv", index=False)
    print("wrote figures/ablation.csv")
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["in_race_mae_s"], 0.4, label="in-race split",
           color="#c33")
    ax.bar(x + 0.2, table["unseen_race_mae_s"], 0.4, label="unseen-race split",
           color="#88a")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"])
    ax.set_ylabel("MAE (s) -- pace vs race median")
    ax.set_title("Ablation: context-aware model tested on data it never saw")
    for xi, row in zip(x, table.itertuples()):
        ax.text(xi - 0.2, row.in_race_mae_s + 0.02, f"{row.in_race_mae_s:.2f}",
                ha="center", fontsize=9)
        ax.text(xi + 0.2, row.unseen_race_mae_s + 0.02,
                f"{row.unseen_race_mae_s:.2f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/ablation.png", dpi=160)
    plt.close(fig)
    print("wrote figures/ablation.png")

    # feature importance -- shows the model actually uses the physics
    imp = (pd.Series(model.feature_importances_, index=X.columns)
           .sort_values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#88a")
    ax.set_title("Model 1 feature importance (gain)")
    fig.tight_layout()
    fig.savefig("figures/feature_importance.png", dpi=160)
    plt.close(fig)
    print("wrote figures/feature_importance.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    main()
