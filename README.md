# Microstructures: Data Generation and Forward Learning

This repository contains an end-to-end pipeline for:

1. generating 2D fiber-reinforced composite RVEs with FEniCSx,
2. preprocessing those outputs into learning-ready `X` and `Y` tensors, and
3. training forward surrogate models that map microstructure inputs to elastic response outputs.

The codebase is organized into two top-level modules:

- `data_generation/`: FEniCSx simulation, preprocessing, and visualization
- `training/`: dataset utilities, neural network models, training, and evaluation


## Problem Setup

The generated samples solve a small-strain elasticity problem for fiber-reinforced composites:

- matrix and fiber phases are defined by `E` and `nu`
- one edge is pinned in `x`
- the bottom-left corner is pinned in `y`
- the opposite edge is given a displacement in `x`

The main outputs are:

- local displacement fields: `ux`, `uy`
- local strain fields: `epsilon_xx`, `epsilon_yy`, `epsilon_xy`
- local stress fields: `sigma_xx`, `sigma_yy`, `sigma_xy`
- global averaged response: `avg_epsilon_*`, `avg_sigma_*`


## Environment

Use the conda environment `microstructures`.

Create it on a new machine with:

```bash
conda env create -f environment.yaml
conda activate microstructures
```

If the environment already exists and you want to sync it to the repo spec:

```bash
conda env update -f environment.yaml --prune
conda activate microstructures
```

Recommended package set used by the current code:

- `fenics-dolfinx`
- `gmsh`
- `python-gmsh`
- `pytorch`
- `torchvision`
- `scikit-learn`
- `tqdm`

Example check:

```bash
conda run -n microstructures python -c "import dolfinx, gmsh, torch; print(dolfinx.__version__, gmsh.__version__, torch.__version__)"
```

The environment specification is stored in:

- [environment.yaml](/Users/shreyanganguly/Python/microstructures/environment.yaml:1)


## Repository Layout

```text
microstructures/
├── AGENTS.md
├── README.md
├── data_generation/
│   ├── generate_fenicsx_rve_dataset.py
│   ├── preprocess_microstructure_inputs.py
│   ├── preprocess_material_field_inputs.py
│   ├── preprocess_fenicsx_targets.py
│   └── visualize_fenicsx_samples.py
└── training/
    ├── datasets.py
    ├── models.py
    ├── train_forward_surrogate.py
    └── evaluate_forward_surrogate.py
```


## 1. Data Generation

Generate a batch of FEniCSx samples:

```bash
conda run -n microstructures python data_generation/generate_fenicsx_rve_dataset.py \
  num=20 \
  start_seed=1 \
  cpus=4 \
  L=150 \
  N_fibers=20 \
  Vf=0.4 \
  matrix_E=3.2e9 \
  matrix_nu=0.35 \
  fiber_E=87.0e9 \
  fiber_nu=0.20 \
  applied_strain_x=0.001 \
  mesh_size=2.0 \
  write_xdmf=false \
  output_dir=outputs/fenicsx_batch
```

Generate elliptical fibers instead of circles:

```bash
conda run -n microstructures python data_generation/generate_fenicsx_rve_dataset.py \
  num=20 \
  start_seed=1 \
  cpus=4 \
  L=150 \
  N_fibers=20 \
  Vf=0.4 \
  fiber_shape=ellipse \
  fiber_aspect_ratio=2.0 \
  random_fiber_angle=true \
  matrix_E=3.2e9 \
  matrix_nu=0.35 \
  fiber_E=87.0e9 \
  fiber_nu=0.20 \
  applied_strain_x=0.001 \
  mesh_size=2.0 \
  write_xdmf=false \
  output_dir=outputs/fenicsx_ellipse_batch
```

Ellipse-specific knobs:

- `fiber_shape=circle|ellipse`
- `fiber_aspect_ratio` sets the semi-major / semi-minor ratio for ellipses and must be `>= 1.0`
- `fiber_angle_deg` sets a fixed in-plane angle in degrees when `random_fiber_angle=false`
- `random_fiber_angle=true` samples a separate angle in `[0, 180)` for each fiber

The total fiber area fraction `Vf` is preserved for both circles and ellipses.

Per sample, the script writes:

- `job_s<seed>/sample_data.npz`
- `job_s<seed>/metadata.json`
- `job_s<seed>/solution.xdmf` only if `write_xdmf=true`
- `job_s<seed>/solution.h5` only if `write_xdmf=true`

The batch folder also contains:

- `batch_summary.json`

Important runtime notes:

- `cpus` parallelizes across independent sample seeds using worker processes.
- `write_xdmf=false` is the recommended production setting for training-data generation.
- the generator prints progress bars and per-sample timings
- `batch_summary.json` includes a `timing_summary` section and total wall time
- the preprocessing scripts read geometry from FEniCSx `metadata.json`, so ellipse batches can be rasterized without extra flags


## 2. Visualize Generated Samples

Render summary figures from one sample or a whole batch:

```bash
conda run -n microstructures python data_generation/visualize_fenicsx_samples.py \
  --input outputs/fenicsx_batch \
  --output_dir outputs/fenicsx_batch/figures
```

Useful options:

- `--seed 7`
- `--deformation_scale 10.0`
- `--show 1`

The visualization script also shows a progress bar while rendering batches.


## 3. Build Learning Inputs `X`

There are two main input choices for forward learning.

### Option A: Binary Microstructure Images

This creates `X` from the phase layout only.

```bash
conda run -n microstructures python data_generation/preprocess_microstructure_inputs.py \
  --input outputs/fenicsx_batch \
  --img_size 128 \
  --channels 1 \
  --L 150 \
  --N_fibers 20 \
  --Vf 0.4 \
  --output outputs/X_microstructure.npy \
  --meta_out outputs/X_microstructure_meta.json
```

When `--input` points to FEniCSx sample folders, this script reads fiber geometry from each sample's `metadata.json`. That includes ellipse axes and orientations when present.

Output tensor shape:

- `(H, W, C, N)`

Typical channels:

- `['fiber']`
- or `['matrix', 'fiber']`

### Option B: Material-Field Images

This creates `X` with per-pixel material channels.

```bash
conda run -n microstructures python data_generation/preprocess_material_field_inputs.py \
  --input outputs/fenicsx_batch \
  --representation material \
  --material_source fenicsx \
  --img_size 128 \
  --L 150 \
  --N_fibers 20 \
  --Vf 0.4 \
  --output outputs/X_material.npy \
  --meta_out outputs/X_material_meta.json
```

As with the binary preprocessor, FEniCSx inputs use the stored sample geometry, so circular and elliptical batches both work.

Typical channels:

- `['E_field', 'nu_field']`


## 4. Build Learning Targets `Y`

The target preprocessor reads the FEniCSx sample folders and emits:

- field tensors: `(H, W, C, N)`
- global-response tensors: `(C, N)`

Example:

```bash
conda run -n microstructures python data_generation/preprocess_fenicsx_targets.py \
  --input outputs/fenicsx_batch \
  --representation both \
  --field_set stress_strain \
  --img_size 128 \
  --field_output outputs/Y_fields.npy \
  --global_output outputs/Y_global.npy \
  --meta_out outputs/Y_meta.json
```

Supported field sets:

- `stress`
- `strain`
- `stress_strain`
- `displacement`
- `all`

Example field channels for `stress_strain`:

- `['epsilon_xx', 'epsilon_yy', 'epsilon_xy', 'sigma_xx', 'sigma_yy', 'sigma_xy']`

Global channels:

- `['avg_epsilon_xx', 'avg_epsilon_yy', 'avg_epsilon_xy', 'avg_sigma_xx', 'avg_sigma_yy', 'avg_sigma_xy']`


## 5. Train the Forward Surrogate

The training script supports three tasks:

- `fields`: predict field outputs only
- `global`: predict global-response outputs only
- `both`: multitask prediction

Example multitask run:

```bash
conda run -n microstructures python training/train_forward_surrogate.py \
  --x outputs/X_microstructure.npy \
  --x_meta outputs/X_microstructure_meta.json \
  --y_fields outputs/Y_fields.npy \
  --y_global outputs/Y_global.npy \
  --y_meta outputs/Y_meta.json \
  --task both \
  --out_dir outputs/training_runs/forward_surrogate \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-3 \
  --val_fraction 0.2 \
  --base_channels 32
```

Saved artifacts:

- `best_model.pt`
- `last_model.pt`
- `training_history.json`
- `run_summary.json`

The training script shows:

- epoch-level progress
- batch-level progress within train/validation loops
- per-epoch train/validation losses


## 6. Evaluate a Trained Model

```bash
conda run -n microstructures python training/evaluate_forward_surrogate.py \
  --checkpoint outputs/training_runs/forward_surrogate/best_model.pt \
  --x outputs/X_microstructure.npy \
  --y_fields outputs/Y_fields.npy \
  --y_global outputs/Y_global.npy \
  --split val \
  --output outputs/training_runs/forward_surrogate/eval_metrics.json
```

Reported metrics:

- field MSE / MAE
- global-response MSE / MAE

The evaluation script also shows a batch-level progress bar.


## Typical End-to-End Workflow

### Binary microstructure input + multitask target

```bash
conda run -n microstructures python data_generation/generate_fenicsx_rve_dataset.py \
  num=100 start_seed=1 cpus=8 L=150 N_fibers=20 Vf=0.4 mesh_size=2.0 write_xdmf=false output_dir=outputs/fenicsx_batch

conda run -n microstructures python data_generation/preprocess_microstructure_inputs.py \
  --input outputs/fenicsx_batch \
  --img_size 128 \
  --channels 1 \
  --L 150 --N_fibers 20 --Vf 0.4 \
  --output outputs/X.npy \
  --meta_out outputs/X_meta.json

conda run -n microstructures python data_generation/preprocess_fenicsx_targets.py \
  --input outputs/fenicsx_batch \
  --representation both \
  --field_set stress_strain \
  --img_size 128 \
  --field_output outputs/Y_fields.npy \
  --global_output outputs/Y_global.npy \
  --meta_out outputs/Y_meta.json

conda run -n microstructures python training/train_forward_surrogate.py \
  --x outputs/X.npy \
  --x_meta outputs/X_meta.json \
  --y_fields outputs/Y_fields.npy \
  --y_global outputs/Y_global.npy \
  --y_meta outputs/Y_meta.json \
  --task both \
  --out_dir outputs/training_runs/forward_surrogate \
  --epochs 50 \
  --batch_size 8
```

### Elliptical microstructure input + multitask target

```bash
conda run -n microstructures python data_generation/generate_fenicsx_rve_dataset.py \
  num=100 start_seed=1 cpus=4 L=150 N_fibers=20 Vf=0.4 \
  fiber_shape=ellipse fiber_aspect_ratio=2.0 random_fiber_angle=true \
  mesh_size=2.0 write_xdmf=false output_dir=outputs/fenicsx_ellipse_batch

conda run -n microstructures python data_generation/preprocess_microstructure_inputs.py \
  --input outputs/fenicsx_ellipse_batch \
  --img_size 128 \
  --channels 1 \
  --L 150 --N_fibers 20 --Vf 0.4 \
  --output outputs/X_ellipse.npy \
  --meta_out outputs/X_ellipse_meta.json

conda run -n microstructures python data_generation/preprocess_fenicsx_targets.py \
  --input outputs/fenicsx_ellipse_batch \
  --representation both \
  --field_set stress_strain \
  --img_size 128 \
  --field_output outputs/Y_ellipse_fields.npy \
  --global_output outputs/Y_ellipse_global.npy \
  --meta_out outputs/Y_ellipse_meta.json

conda run -n microstructures python training/train_forward_surrogate.py \
  --x outputs/X_ellipse.npy \
  --x_meta outputs/X_ellipse_meta.json \
  --y_fields outputs/Y_ellipse_fields.npy \
  --y_global outputs/Y_ellipse_global.npy \
  --y_meta outputs/Y_ellipse_meta.json \
  --task both \
  --out_dir outputs/training_runs/forward_surrogate_ellipse \
  --epochs 50 \
  --batch_size 8
```

The preprocessing and target-generation steps are unchanged for ellipse batches because they read the stored fiber geometry from the generated FEniCSx sample metadata.


## Data Contracts

### Generated sample folder

Each `job_s<seed>/sample_data.npz` contains arrays such as:

- `points`
- `cells`
- `cell_tags`
- `ux`, `uy`
- `epsilon_xx`, `epsilon_yy`, `epsilon_xy`
- `sigma_xx`, `sigma_yy`, `sigma_xy`
- `E`, `nu`
- `fiber_centers`, `fiber_radius`
- `fiber_axes`, `fiber_angles_deg`

Each `metadata.json` contains:

- generation parameters
- fiber geometry metadata (`fiber_shape`, `fiber_centers`, `fiber_axes`, `fiber_angles_deg`)
- averaged stress/strain summaries
- mesh sizes
- seed and provenance

Each `batch_summary.json` contains:

- `samples_written`
- `failed_seeds`
- `wall_time_seconds`
- `timing_summary`
- the fixed parameter set used for the batch

### Learning tensors

`X` tensors:

- shape `(H, W, C, N)`

`Y_fields` tensors:

- shape `(H, W, C, N)`

`Y_global` tensors:

- shape `(C, N)`


## Notes

- The current model is a compact U-Net-style surrogate with an optional global head.
- The training script normalizes `X`, `Y_fields`, and `Y_global` from the training split only.
- The evaluation script uses the saved normalization statistics from the checkpoint.
- Small smoke tests may produce meaningless losses; use larger datasets for real training.
- For large production batches, prefer `write_xdmf=false` unless you explicitly need visualization files.
- For large datasets, the safest speedup is increasing `cpus`, since samples are independent and quality is unchanged.


## Recommended Next Steps

- choose whether `X` should be binary phase maps or material fields
- choose whether `Y` should be fields, global response, or both
- generate a larger dataset with parameter sweeps over morphology and material contrast
- train separate baselines for:
  - `X -> Y_fields`
  - `X -> Y_global`
  - `X -> (Y_fields, Y_global)`
