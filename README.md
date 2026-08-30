# Inchident

Counterfactual F1 tyre-degradation modelling. Within a single stint tyre age and
fuel load are perfectly collinear, so a naive "lap time vs tyre age" curve is
meaningless. Inchident pools many stints across many races, models lap time from
tyre age **plus context** (fuel, track evolution, thermal, compound, circuit,
traffic), then sweeps only `TyreLife` with everything else held fixed to recover
the true degradation curve `D(a) = P_a - P_0`.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Pipeline (run in order, verify each step)

```bash
# 1. Provisioning -- the ONLY networked script. First run is a long download.
.venv/bin/python -m src.build_dataset --smoke     # 2 races, sanity check
#    confirm data/stints.parquet has TyreLife / Compound / TrackTemp non-null
.venv/bin/python -m src.build_dataset             # full RACES list

# 2. Features + the two load-bearing figures. STOP and look at exp1.
.venv/bin/python -m src.experiments

# 3. Interpretable OLS: coefficient table, fuel-prior check, slope progression.
.venv/bin/python -m src.regression
#    per-circuit + temperature degradation coefficients
.venv/bin/python -m src.degradation
#    then the narrative figures for the idea-submission document
.venv/bin/python -m src.deck_figures

# 4. Ablation + first Inchident curve. Model 1 must beat Model 0.
.venv/bin/python -m src.model

# 5. Demo
.venv/bin/streamlit run app.py
```

## Layout

| Path | Role |
| --- | --- |
| `src/build_dataset.py` | FastF1 -> `data/stints.parquet` (networked, cached) |
| `src/features.py` | fuel / track-evo / thermal proxies |
| `src/experiments.py` | Experiments 0-6, one figure each -> `figures/` |
| `src/regression.py` | Interpretable OLS pace model: coefficients, fuel-prior check, slope progression, `inchident_curve` |
| `src/degradation.py` | Per-circuit + temperature-modulated degradation coefficients -> `deg_*.png` |
| `src/deck_figures.py` | Narrative figures (`deck_*.png`) for `docs/idea-submission.*` |
| `src/model.py` | Model 0 vs Model 1 ablation + counterfactual curve |
| `docs/idea-submission.md` | The 11-slide idea-submission document (HTML artifact alongside) |
| `docs/degradation-estimation.html` | Companion note: degradation coefficient as the estimand, data + regs grounding |
| `app.py` | Streamlit live-slider demo |
| `web/` | Static site (Vercel): landing page, both docs, and `dashboard.html` — the demo curve recomputed client-side from `web/coefficients.json` |

## Static site

`web/` is a zero-config static deploy. `web/dashboard.html` reproduces the
Streamlit demo's four controls (circuit / compound / track temp / tyre age) and
recomputes `D(a)` in the browser from the fitted coefficients in
`web/coefficients.json` (regenerate those from `src.regression` + `src.degradation`).

```bash
cd web && vercel deploy --prod
```

Live: <https://inchident-pink.vercel.app>

## Caveats to state in the deck

- `FUEL_S_PER_KG = 0.03` is a published rule of thumb, not a measurement. The
  model calibrates around it; we do **not** claim to know the fuel load.
- `Position` is a rough dirty-air proxy until gap/interval telemetry is loaded.
- Dry laps only, green-flag only, in/out laps removed, per-stint IQR filtered.
