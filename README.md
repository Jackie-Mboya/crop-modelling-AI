# crop-modelling-AI

**A process-based crop model × machine learning workflow for maize yield prediction and agronomic decision support in sub-Saharan Africa.**

Prototype developed for the IITA "Integrated Crop Modelling and AI Expert" consultancy assessment. The workflow couples a process-based crop simulation engine (native APSIM, with a physiologically calibrated fallback) with a two-stage machine learning layer, an emulator that reproduces process-model behaviour and a bias-correction model that reconciles simulated yield against field-observed yield, orchestrated through a lightweight agentic decision-support pipeline.

---

## 1. Motivation and scope

Process-based crop models such as APSIM and DSSAT remain the standard for mechanistic yield simulation, but they are computationally expensive to run at scale and cannot, on their own, correct for the systematic gap between simulated and field-observed yield that arises from management, pest/disease pressure, and other unmodelled factors. This repository demonstrates a workflow that:

1. Runs a process-based maize model across defined agronomic scenarios (Step 1–2).
2. Generates a synthetic-but-structured simulation dataset spanning realistic ranges of the key agronomic drivers of yield (Step 3).
3. Trains **two distinct ML layers** on top of the process model output, an emulator (Option A) and a bias corrector against field observations (Option B), rather than treating "ML refinement" as a single undifferentiated step (Step 4).
4. Evaluates both layers with standard regression metrics and diagnostic plots (Step 5).
5. Wraps the whole pipeline in a minimal agentic controller that composes deterministic tool calls into a single advisory output (Step 4, Option D).

The design intent is to sketch how such a workflow could plug into an operational agronomic Decision Support Tool: the process model supplies mechanistic grounding, the emulator supplies cheap scalability, and the bias corrector supplies the closest approximation to what a farmer would actually observe in the field.

## 2. Repository structure

```
crop-modelling-AI/
├── main.py                        orchestrates the full pipeline, Steps 1–5
├── requirements.txt
├── config/
│   └── crop_model_config.json     APSIM binary path, base .apsimx file, location metadata
├── src/
│   ├── apsim_driver.py            NativeAPSIMDriver: runs APSIM if available, else calibrated fallback
│   ├── ml_refinement.py           MLRefinementLayer: emulator (Option A) + bias corrector (Option B)
│   ├── evaluate.py                diagnostic plots (parity, N response curve)
│   └── agentic_workflow.py        AgronomicAgent: tool-calling advisory pipeline (Option D)
├── data/
│   └── simulation_dataset.csv     n=100 simulated scenarios (Step 3 output)
└── outputs/
    ├── scenario_comparison.csv    Step 2 fertilizer-scenario results
    ├── evaluation_metrics.csv     Step 5 metrics table
    └── evaluation_plots.png       Step 5 diagnostic figure
```

## 3. Crop model layer: `src/apsim_driver.py`

`NativeAPSIMDriver` is written to call a real, locally installed APSIM Next Generation binary (`Models.exe`) against a base `.apsimx` maize simulation file, with the binary path and file location read from `config/crop_model_config.json`. Where the APSIM executable and simulation file are not present, with no APSIM installation available, the driver falls back to a physiologically parameterised proxy of APSIM's own maize yield response, rather than an arbitrary placeholder function:

- A **nitrogen response** term using a saturating exponential of applied N (asymptoting toward potential yield, consistent with APSIM's N-uptake and biomass partitioning behaviour).
- A **water response** term using a logistic function of seasonal rainfall, approximating the sigmoidal yield–water relationship produced by APSIM's soil water balance.
- A **sowing-date penalty**, reflecting the yield loss APSIM typically simulates for maize sown outside the optimal March window in this agro-ecological zone.
- A fixed **potential yield ceiling** (8,500 kg/ha) representative of well-managed hybrid maize under near-optimal conditions in western Kenya.

This is a disclosed simplification, not a substitute for a calibrated APSIM run: the `run_single_simulation` method is structured so that supplying a real `apsim_binary_path` and `.apsimx` file, and installing `apsimNGpy` per `requirements.txt`, routes execution through actual APSIM output with no other code changes required. The fallback exists purely to keep the pipeline runnable end-to-end in the absence of a local APSIM installation, and the yield magnitudes it produces (baseline ≈3.6 t/ha, moderate-N ≈6.2 t/ha, high-N ≈7.1 t/ha for the Step 2 scenarios below) sit within the range reported in Kenyan hybrid-maize agronomy trials.

**Location and crop.** Western Kenya, Zone 1 (`config/crop_model_config.json`), maize, sown 15 March in the baseline scenarios.

## 4. Step 2: Fertilizer scenario comparison

Three nitrogen scenarios at fixed sowing date (15 March) and rainfall (550 mm):

| Scenario | N rate (kg/ha) | Simulated yield (kg/ha) | Simulated yield (t/ha) |
|---|---|---|---|
| Baseline (no fertilizer) | 0 | 3,613.2 | 3.61 |
| Moderate N | 60 | 6,152.2 | 6.15 |
| High N | 120 | 6,520.3 | 6.52 |

The response shows the expected diminishing-returns pattern of a saturating N-uptake function: the yield gain from 0 to 60 kg N/ha (+2,623 kg/ha) is more than double the gain from 60 to 120 kg N/ha (+891 kg/ha). Output: `outputs/scenario_comparison.csv`.

## 5. Step 3: Simulation dataset

`generate_simulation_dataset()` produces n=100 records varying three primary agronomic factors, **rainfall** (280–850 mm), **nitrogen rate** (0–160 kg/ha), and **sowing date** (four dates spanning 10 March–20 April), with **soil organic carbon** (0.8–2.2%) as a fourth covariate. For each record the driver also constructs a synthetic **field-observed yield**, offset from the simulated yield by a stochastic management-efficiency factor (0.75–0.92), a rainfall-conditional pest/disease loss term (higher under high-rainfall, high-pest-pressure conditions), and Gaussian noise. This observed/simulated pairing is what makes the Option B bias-correction model trainable: without a disclosed, structured gap between the two, there is nothing for a bias corrector to learn. Output: `data/simulation_dataset.csv`.

## 6. Step 4: ML refinement layer (`src/ml_refinement.py`)

`MLRefinementLayer` trains two independently evaluated random-forest regressors (100 trees each) on an 80/20 split of the Step 3 dataset:

- **Option A - ML Emulator.** Trained on `[rainfall_mm, nitrogen_kg_ha, soc_pct, sowing_code]` to reproduce `simulated_yield_kg_ha`. Purpose: approximate the process model cheaply enough to scale predictions across many locations or grid cells without re-running APSIM for each one.
- **Option B - Bias Corrector.** Trained on the same four covariates plus the simulated yield itself, to predict `observed_yield_kg_ha`. Purpose: correct the process model's output toward what would actually be observed in the field, which is the metric that ultimately matters for a farmer-facing advisory tool.

Running both, rather than only an emulator, was a deliberate choice: an emulator that faithfully reproduces the process model is a scaling tool, not a validation tool, since it is only ever as accurate as the process model it imitates. The bias corrector is the layer that actually closes the sim-to-real gap.

### Step 5: Evaluation

| Layer | RMSE (kg/ha) | MAE (kg/ha) | R² | MBE (kg/ha) | MAPE (%) |
|---|---|---|---|---|---|
| Option A: ML Emulator (vs. APSIM sim) | 464.80 | 388.42 | 0.930 | -61.21 | 8.96 |
| Option B: Bias Corrector (vs. field obs) | 241.73 | 195.75 | 0.972 | +27.63 | 4.68 |

The bias corrector outperforms the pure emulator on every metric, expected, since it is given the simulated yield as an additional input and only needs to learn the (comparatively lower-variance) residual structure, rather than the full process-model response surface from four covariates alone. The near-zero mean bias error for both layers indicates neither is systematically over- or under-predicting; the emulator's slightly larger RMSE/MAPE reflects the harder task of reconstructing the process model's nonlinear response purely from inputs.

Diagnostics in `outputs/evaluation_plots.png`:
- **Emulator parity plot** - predicted vs. APSIM-simulated yield on the test set, against the 1:1 line.
- **Nitrogen response curve** - ML-predicted yield across the full 0–160 kg N/ha range at fixed rainfall (580 mm), overlaid with test-set predictions, to visually confirm the model has learned a physiologically sensible (saturating, monotonic) N response rather than an artefact of the training sample.

**Does the ML layer reproduce or improve on the crop model?** The emulator reproduces it well (R²=0.930) but cannot exceed the process model's own accuracy, since it is trained only on the process model's output. The bias corrector is the layer that genuinely improves on the process model against ground truth (R²=0.972 vs. field-observed yield), because it is explicitly trained to correct the process model's systematic error rather than imitate it.

**What field data would be needed for proper validation?** This prototype's "observed" yield is synthetic, constructed to be structurally learnable rather than measured. Operational deployment would require georeferenced, season- and plot-level yield records with matched planting date, fertilizer rate, and soil sampling, alongside station or satellite-derived rainfall and temperature for the corresponding fields, so the bias corrector is trained against real management and environmental variance rather than a synthetic proxy for it.

## 7. Step 4, Option D - Agentic advisory workflow (`src/agentic_workflow.py`)

`AgronomicAgent` composes the process model and ML layers into a single advisory pipeline via discrete tool calls: `get_weather_data()` --> `get_soil_data()` --> process-model simulation --> ML emulator prediction --> formatted advisory. Example output for Western Kenya, 85 kg N/ha, sown 15 March:

```
• APSIM Process Model Prediction : 6,845.8 kg/ha (6.85 t/ha)
• ML Emulator Surrogate Prediction: 6,284.8 kg/ha (6.28 t/ha)
• Agronomic Action: Apply Nitrogen split (50% basal at planting, 50% top-dressed at V6 stage).
```

**Where an LLM belongs in this pipeline, and where it does not.** Every step up to and including the yield predictions above is deterministic, numerically reproducible, and physically grounded, `get_weather_data`, `get_soil_data`, the APSIM/fallback simulation, and the ML predictions must remain plain function/model calls, never LLM-generated, because agronomic recommendations need to be auditable and reproducible run-to-run. The one place an LLM adds legitimate value is turning the structured numeric output above into fluent, context-appropriate natural language for an extension officer or farmer, and even there, it should be constrained to phrase and explain the numbers it is given, never to generate or alter a yield figure itself. The current `run_advisory_pipeline` implementation uses a deterministic string template for exactly this reason; substituting an LLM call at that single point (with the structured dict as its only source of numeric truth) is the natural extension for a production system.

## 8. How to run
Clone the repository using "git clone https://github.com/Jackie-Mboya/crop-modelling-AI.git"
Open using IDE environment like Visual Studio Code (VS code)
Open terminal then run the following
```bash
pip install -r requirements.txt
python main.py
```

Note: you can also install a virtual machine using `python -m venv venv`

This executes Steps 1 - 5 end to end: scenario comparison --> simulation dataset generation --> ML training and evaluation --> diagnostic plots --> agentic advisory demo. All outputs are written to `outputs/` and `data/`.

To run against real APSIM instead of the fallback: install [`apsimNGpy`](https://github.com/APSIMInitiative/ApsimNGpy) (listed in `requirements.txt`), install APSIM Next Generation locally, and update `apsim_binary_path` and `base_apsimx_file` in `config/crop_model_config.json` to point to a valid installation and baseline `.apsimx` maize simulation file.

## 9. Key assumptions and limitations

- **APSIM fallback.** In the absence of a local APSIM installation, yield is generated by a physiologically parameterised proxy, not a calibrated APSIM run. Absolute yield levels should be read as agronomically plausible, not site-validated forecasts.
- **Synthetic weather and soil inputs.** Rainfall, nitrogen, sowing date, and SOC ranges are sampled to span realistic agronomic conditions for the target zone; they are not drawn from observed station or survey data.
- **Synthetic observed yield.** The Option B bias corrector is validated against a constructed, not measured, observed-yield variable, so its reported skill (R²=0.972) reflects how well it recovers a known synthetic bias structure, not real-world predictive skill.
- **Single location and crop.** The workflow is demonstrated for maize in one agro-ecological zone (Western Kenya); scaling to multiple crops/locations would require either additional APSIM configuration files per zone or the spatial-scaling extension (Option C) not implemented here.
- **Random-forest models are not extrapolation-safe.** Both ML layers are trained on n=100 records; predictions for input combinations well outside the sampled ranges (e.g., N rates near the upper or lower bound combined with extreme rainfall) should be treated as unreliable.

## 10. License

MIT, see `LICENSE`.
