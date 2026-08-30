# Inchident — Interpretable Regression + Idea-Submission Document

**Date:** 2026-08-29
**Status:** Approved (brainstorming)

## Goal

Add an interpretable linear-regression layer to the existing Inchident pipeline
and produce an idea-submission document (Markdown + HTML artifact) that walks
the reader from the naive "tyre age → degradation" intuition, through real F1
data that breaks it, to a counterfactual regression estimate of true tyre
degradation validated on held-out races.

The regression must "sit truly correct with the F1 data": transparent
coefficients, honest reporting of what is identified vs assumed, and validation
on races the model never saw.

## Non-goals

- No new network calls. Uses the existing `data/stints.parquet` (13 races).
- Not replacing the existing XGBoost `src/model.py` — it stays as supporting
  material. The linear regression is the new headline evidence.
- No temporal / multimodal / PINN modelling — that is explicitly "future work"
  in the document, not built here.

## The statistical core

Within a single stint, tyre age and fuel load are near-perfectly collinear
(both linear in lap number), so `corr(TyreLife, FuelKg)` inside one stint
≈ −0.99. A single stint cannot separate tyre degradation from fuel-burn gain.

Pooling stints that start at different race laps breaks the collinearity: a
stint starting on lap 2 and one starting on lap 34 sit at very different fuel
loads at the same tyre age. Pooled across 13 races the correlation drops far
enough for OLS to attribute variance correctly. This identifiability argument
is the intellectual point of the submission and is shown as a figure.

## Components

### `src/regression.py` (new)

Plain OLS via `statsmodels.formula.api.ols`. Fully inspectable.

Formula:

```
LapTimeS ~ C(Event)                     # circuit fixed effects
         + TyreLife + I(TyreLife**2)    # degradation: linear + curvature
         + FuelKg                       # fuel effect, ESTIMATED
         + TrackEvoIdx
         + TrackTemp
         + C(Compound)
```

Data filter: reuse what `build_dataset.py` already produced (green, dry, racing
laps, per-stint IQR cleaned). Additionally drop `TyreLife > 40` (already done in
build) and require `FreshStint` is not imposed here — all racing laps are used
so fuel varies widely at each tyre age.

Public functions:

- `fit_pace_model(df) -> statsmodels results` — the full fit.
- `fit_naive_model(df) -> results` — `LapTimeS ~ C(Event) + TyreLife + C(Compound)`
  (Event kept so both models are scored on the same target scale; the naive one
  just lacks the physics covariates).
- `coefficient_table(results) -> pd.DataFrame` — term, estimate, 95% CI, p.
- `inchident_curve(results, context_row, max_age=25) -> (ages, penalty)` —
  counterfactual: sweep TyreLife, hold everything else fixed,
  `penalty(a) = predicted(a) - predicted(0)`.
- `groupkfold_mae(df, fit_fn, k=5) -> float` — MAE by RaceId folds.
- `fuel_prior_check(results) -> dict` — estimated `FuelKg` coef and s/kg vs the
  0.03 s/kg prior from `features.py`. If the estimate is unstable / wrong-signed,
  the module also supports `fit_pace_model(df, impose_fuel=True)` which subtracts
  `FuelEffectS` as a fixed offset and drops `FuelKg` from the RHS. The document
  reports whichever path is used and why.

Run as `python -m src.regression` → prints the coefficient table, the fuel-prior
check, and naive-vs-context GroupKFold MAE; writes `figures/deck_coefficients.png`.

### `src/deck_figures.py` (new)

Generates the narrative figures into `figures/`. Reads
`src.features.load_features()` and `src.regression`.

| File | Slide | Content |
|---|---|---|
| `deck_raw_pace.png` | 3 | Bahrain 2023: `PaceVsRaceMedian` per lap for ~4 representative fresh stints, coloured by stint, x = tyre age. Shows pace falling mid-stint as the tyre ages, and stint-to-stint inconsistency. |
| `deck_collinearity.png` | 4-5 | Bar / annotated plot: `corr(TyreLife, FuelKg)` within-stint (median over stints, ≈−0.99) vs pooled across all laps (≈−0.3). |
| `deck_decomposition.png` | 7 | One long Bahrain 2023 stint: observed pace vs start-of-stint, decomposed into fuel (from the fitted `FuelKg` coef), track-evo, and tyre-deg components (stacked area or lines). |
| `deck_naive_vs_context.png` | 8 | Left: raw `PaceVsRaceMedian` binned by tyre age, pooled (flat / noisy / near-zero slope). Right: regression counterfactual Inchident curve (clean, rising). Shared y-axis. |
| `deck_validation.png` | 9 | 2 panels (Bahrain 2023, Spain 2023): predicted Inchident curve from a model fit on the *other 12 races* vs the featured stint's actual fuel-corrected pace (re-centred on first 3 laps). |
| `deck_coefficients.png` | 7 | Coefficient plot (point + 95% CI) for TyreLife, TyreLife^2, FuelKg, TrackEvoIdx, TrackTemp. Written by `regression.py`. |

Featured stints: pick the longest `FreshStint` MEDIUM (Spain) / HARD (Bahrain,
few mediums there) stint in the race, tie-break by lowest lap-time variance.

Run as `python -m src.deck_figures`.

### `docs/idea-submission.md` (new)

The 11-slide structure supplied by the user, softened framing (no "mathematically
exact", "highly accurate", "precise optimization" — instead "we propose",
"we demonstrate a first version", "we validate whether"). Real numbers from
`regression.py` filled in: estimated fuel s/kg, tyre-deg per lap, held-out MAEs,
the two validation overlays. Figures referenced from `figures/`.

### `docs/idea-submission.html` (new → Artifact)

Same content as a clean single-page scrollable deck, figures embedded as
`<img>` referencing local paths for the repo copy; for the published Artifact
the figures are inlined as base64 data URIs (Artifact CSP blocks external/local
images). Theme-aware, responsive. Published via the Artifact tool.

## Edits to existing files

- `requirements.txt` — add `statsmodels>=0.14`.
- `README.md` — add the regression + deck steps to the pipeline order and the
  layout table.
- `Makefile` — add `deck` target: `python -m src.regression && python -m src.deck_figures`.

## Untouched

`src/build_dataset.py`, `data/stints.parquet`, `data/cache/`, `src/model.py`,
`src/experiments.py`, `src/features.py`, `app.py`.

## Validation / success criteria

1. `python -m src.regression` runs, prints a coefficient table with finite CIs.
2. Estimated `FuelKg` coefficient is negative (more fuel → slower, i.e. positive
   s/kg) and within ~2x of the 0.03 s/kg prior — OR the `impose_fuel=True` path
   is used and the document says so.
3. `TyreLife` linear coefficient is positive (older → slower) after controlling
   for fuel and evolution.
4. Context model GroupKFold MAE < naive model GroupKFold MAE.
5. `deck_naive_vs_context.png`: right panel slope clearly steeper / cleaner than
   left.
6. `deck_validation.png`: predicted curve tracks the observed fuel-corrected
   points on both held-out races (qualitative — within a few tenths).
7. All six deck figures written; document builds with real numbers; HTML
   artifact publishes.

## Risks

- **Fuel coef unidentified / wrong sign** even pooled → fall back to
  `impose_fuel=True`. Document honestly. (Mitigation already in the design.)
- **Quadratic tyre term unstable** → drop `I(TyreLife**2)`, use linear only.
- **statsmodels not installable on the 3.12 venv** → unlikely (pure-ish, has
  wheels); fallback is `numpy.linalg.lstsq` with manual dummy encoding and
  bootstrap CIs.
- **Validation overlay noisy** for a single stint → average 2-3 stints of the
  same compound in that race for the observed reference.
