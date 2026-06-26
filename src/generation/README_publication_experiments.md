# Publication-level OR-ICU-Ward experiment scripts

These scripts add a publication-facing experiment layer on top of the existing code base.
They do not rename your original algorithm modules. Instead, they keep internal folders for compatibility and map them to manuscript-ready method names during collection and plotting.

## 1. Files added

- `publication_experiment_config.py`  
  Publication sizes, seeds, scenario names, output roots, method-name mapping, and reusable parameters.

- `batch_generate_publication_scenarios.py`  
  Generates nominal and ward-transfer stress scenarios.

- `batch_run_publication_pipeline.py`  
  Runs the core pipeline, optional LNS ablations, and optional integrated MIP benchmarks.

- `collect_publication_results.py`  
  Collects all results and maps internal method names to publication names.

- `plot_publication_results.py`  
  Produces publication-facing plots using the cleaned method names.

## 2. Experiment interpretation

The size parameter `n` is the candidate elective patient pool size for the weekly planning problem. It is not the number of surgeries performed. This is important when using `n=150`.

Recommended scenarios:

- `nominal`: original generated instance.
- `ward_pressure`: ward capacity reduced by 10% on all days.
- `transfer_bottleneck`: ward capacity reduced by 25% on days 3--5.

The `transfer_bottleneck` scenario is designed to test whether explicit ICU-to-ward transfer recourse helps when transfer timing, not just aggregate bed load, becomes binding.

## 3. Required original modules

Place these new scripts in the same folder as your existing modules, including:

- `shehadeh_style_benchmark_generator_v3.py`
- `surgery_schedule_evaluator.py`
- `surgery_baselines.py`
- `surgery_ba_hla_v2.py`
- `surgery_ba_hla_v3.py`
- `surgery_ba_hla_v41.py`
- `stage2_priority_soft_gurobi_repair_v3_fixed.py`
- `stage2_feedback_gurobi_repair_ot_control_fixed.py`
- `adaptive_hp_forced_feedback.py`
- `stage3_icu_ward_blocking_flow_mip_fixed.py`
- `pr_glns_or_icu_ward.py`
- `pr_glns_spiral_downstream_aggressive.py`
- `shehadeh_style_integrated_mip_baseline_v2_fixed.py`
- `batch_run_shehadeh_adaptive_baselines.py` if running integrated MIP benchmarks.

## 4. Suggested run order

### Step 1: generate instances

```bash
python batch_generate_publication_scenarios.py \
  --sizes 50 70 100 150 \
  --seeds 7 11 19 23 29 31 37 41 43 47 \
  --scenarios nominal ward_pressure transfer_bottleneck
```

### Step 2: run the core methods

This runs the initial heuristic, execution-repaired baseline, transfer recourse evaluation, and recourse-feedback repair.

```bash
python batch_run_publication_pipeline.py \
  --sizes 50 70 100 150 \
  --seeds 7 11 19 23 29 31 37 41 43 47 \
  --scenarios nominal ward_pressure transfer_bottleneck
```

### Step 3: run LNS ablations and intensification

```bash
python batch_run_publication_pipeline.py \
  --sizes 50 70 100 150 \
  --seeds 7 11 19 23 29 31 37 41 43 47 \
  --scenarios nominal ward_pressure transfer_bottleneck \
  --run-lns
```

### Step 4: run integrated MIP benchmarks

The adaptive integrated MIP can be expensive. A defensible first pass is to run it for `n=50,70,100` and selected scenarios.

```bash
python batch_run_publication_pipeline.py \
  --sizes 50 70 100 \
  --seeds 7 11 19 23 29 31 37 41 43 47 \
  --scenarios nominal transfer_bottleneck \
  --run-mip-benchmarks
```

Use `--include-free-mip` only if you want the free-admission sensitivity benchmark:

```bash
python batch_run_publication_pipeline.py \
  --sizes 50 70 100 \
  --seeds 7 11 19 23 29 31 37 41 43 47 \
  --scenarios nominal transfer_bottleneck \
  --run-mip-benchmarks --include-free-mip
```

### Step 5: collect results

```bash
python collect_publication_results.py
```

Outputs:

- `publication_tables/publication_results_detailed.csv`
- `publication_tables/publication_results_aggregate.csv`

### Step 6: create figures

```bash
python plot_publication_results.py \
  --input publication_tables/publication_results_detailed.csv
```

Outputs are saved in `publication_figures/`.

## 5. Publication method names

The collector maps internal folders to manuscript-facing names:

| Internal folder | Publication name |
|---|---|
| `02_stage2_v3_volume_first` | Execution-repaired baseline |
| `04_hp_forced_ot_feedback` | Recourse-feedback repair |
| `07_v41_weak_pr_glns` | Unguided LNS control |
| `06_downstream_aggressive_spiral_pr_glns` | Recourse-guided LNS |
| `10_shehadeh_adaptive_access` | Adaptive integrated MIP |
| `10_shehadeh_adaptive_free` | Free-admission integrated MIP |

Do not use the internal names in the manuscript text, tables, or figure labels.

## 6. Recommended paper narrative

The main experimental questions should be stated as:

1. Does transfer-recourse feedback improve an execution-repaired schedule?
2. Does recourse-guided LNS outperform unguided LNS under the same evaluation framework?
3. Under ward-transfer stress, does explicit recourse feedback reduce blocked transfer patient-days?
4. What trade-off appears among high-priority access, execution violations, OR overtime, and downstream patient flow?

Avoid claiming that the heuristic uniformly dominates the integrated MIP. The safer interpretation is that the methods represent different operating regimes and expose the trade-off among access, execution feasibility, overtime, and transfer blocking.
