# MDGCN

Code and data for **Multi-view Dynamic Graph Deep Learning for Surface Defect Identification from 3D Point Cloud**, by Kexin Wu, Xing Wang and Xiaowei Yue, accepted for publication in *IISE Transactions* (UIIE-9011.R3).

The training code, analysis scripts and figure-generation tools are available in this repository. The [reproducibility release](https://github.com/Wukexin1/MDGCN/releases/tag/iise-repro-20261116) provides the prepared inputs, fixed splits, training records and saved predictions.

Download or clone the code, then extract the [inputs and results archive](https://github.com/Wukexin1/MDGCN/releases/download/iise-repro-20261116/MDGCN_inputs_and_results_20261025.zip) into the repository root before running the commands below. Geodesic experiments use the original distance files provided with MVGCN. See [data downloads and extraction](docs/DOWNLOADS.md).

| Material | Link |
|---|---|
| Code repository | [Wukexin1/MDGCN](https://github.com/Wukexin1/MDGCN) |
| Prepared inputs, fixed splits and saved results | [Versioned download](https://github.com/Wukexin1/MDGCN/releases/download/iise-repro-20261116/MDGCN_inputs_and_results_20261025.zip) |
| Original data and geodesic distances | [MVGCN data links and import instructions](docs/DOWNLOADS.md#geodesic-distance-files) |

## Data

The datasets are adapted from [MVGCN](https://github.com/wyn430/MVGCN). Its authors provide the [synthetic data](https://drive.google.com/drive/folders/11d5L1DJomhu_RCGtVBslECfsJv48o7AC) and [real data](https://drive.google.com/drive/folders/1qQh3wgQQpUl0genZ-yklOI1F4FcpuhMS). This release includes the inputs and fixed training, validation and test splits used in our experiments. Use these prepared inputs for reproduction, since the upstream folders contain different preprocessing variants. The full `Front_fusion_30.ply` scan is excluded; the classification experiments use cropped point clouds. File layouts and label meanings are described in [DATA.md](docs/DATA.md).

## Experiments

| Program | Runs | Experiments |
|---|---:|---|
| `code/complete` | 363 | Baseline models; Euclidean and geodesic distance; coordinates and normal vectors; defect fractions; neighborhood size; threshold percentiles |
| `code/uniform` | 36 | MVGCN, MDGCN1 and MDGCN2 using a common surface-variation rule calibrated on normal training point clouds |
| `code/components` | 72 | Standard, hard and soft operators with replace or add fusion on both datasets, at batch sizes 32 and 25 |

The experiments use seeds 24, 42 and 2026 and 250 training epochs. Complete and uniform select by validation overall accuracy; components selects by validation balanced accuracy. Ties are resolved by lower validation cross entropy, then the earliest epoch. Each program retains its own inputs and selection rule.

In complete and uniform, `hard` denotes MDGCN1 and `soft` denotes MDGCN2. The other model names are `pointnet`, `pointnet2`, `dgcnn`, `mvgcn` and, in the controlled-fraction study, `dsrf`. In components, `standard`, `hard` and `soft` choose the local operator, while `replace` and `add` choose fusion. The figure and table mapping is in `analysis/figure_table_mapping.json`.

The saved results cover all 471 runs: 363 complete, 36 uniform and 72 components. The public package includes their configurations, training histories and predictions. Historical checkpoints were verified locally against the run records and are not included in this release. The training programs generate new checkpoints from the supplied inputs and settings. See [release status](docs/RELEASE_STATUS.txt).

## Environment

Training records specify Python 3.8.20 and PyTorch 2.4.1 with CUDA 11.8 on an NVIDIA A100 GPU. The returned server snapshot also records NumPy 1.24.3 and Matplotlib 3.7.5. See [ENVIRONMENT.md](docs/ENVIRONMENT.md) for installation commands, the full current environment record and the distinction between training records and the later snapshot.

## Recalculate the reported results

From the repository root:

```bash
python -m pip install -r requirements-analysis.in
python analysis/reproduce_statistics.py \
  --complete retained_results/complete_return \
  --uniform retained_results/uniform_return \
  --components retained_results/component_return \
  --output new_statistics_check
```

This calculation uses saved predictions and requires no GPU or model weights. It checks the reported statistics and writes per-run results, grouped summaries and a verification report. The completed check covers 471 runs, 75,630 prediction rows, 630 displayed mean/standard-deviation pairs and 30 candidate-region statistics. See `analysis/README_results_only.txt` for the detailed scope.

## Reproduce the figures

```bash
python tools/plot_figures.py --output generated/figures
```

This produces Figures 4 and 6 as PDF and 600 dpi PNG, with the sample IDs and curve values. Figure 4 uses the prepared synthetic inputs; Figure 6 uses 45 saved controlled-fraction runs. Select one figure with `--figure 4` or `--figure 6`. Figures 1–3 and the original scan views in Figure 5 are retained in `figure_assets/`. See [figure documentation](docs/figures.txt).

## Evaluate a saved model checkpoint

List the run IDs and check the required inputs:

```bash
python tools/evaluate_saved.py --list-runs --family complete
python tools/evaluate_saved.py \
  --family complete \
  --run-id surface_hard_xyz_k15_base_rNA_s24_b32_e250 \
  --check-inputs
```

The following example checks a historical checkpoint if you already hold the matching file. Historical checkpoints are not bundled with this release. Place the file at `recovered_checkpoints/complete/runs/<run_id>/selected.pt`, then run:

```bash
python tools/evaluate_saved.py \
  --family complete \
  --run-id surface_hard_xyz_k15_base_rNA_s24_b32_e250 \
  --weights-root recovered_checkpoints \
  --split test \
  --device cuda \
  --output new_inference_check
```

The evaluator verifies the checkpoint and input hashes, uses the recorded settings, and compares the new predictions with the saved results. Use a new output directory for each evaluation. `--split all` includes training, validation and test data; `--checkpoint /path/to/selected.pt` accepts an individual file. The other protocol names are `uniform` and `components`. See [evaluation documentation](docs/saved_evaluation.txt).

Four selected historical models have been evaluated locally on their complete test splits: synthetic PointNet, synthetic MDGCN1, real-data MDGCN2, and a real-data component run with soft aggregation and add fusion at batch size 25. All use seed 24 and coordinate inputs. Their 176 predicted categories agree with the saved results. This CPU check used PyTorch 2.6.0; the training runs used CUDA and PyTorch 2.4.1. Probability differences and measured times are recorded in `verification/inference_summary.json`.

Geodesic runs also need the distance cache in `code/complete/data/surface/geo/` and `code/complete/data/real/geo/`. Import it from the original MVGCN distance files using the instructions in [DOWNLOADS.md](docs/DOWNLOADS.md#geodesic-distance-files). The importer checks the recorded file hashes.

## Train the models

Run each block from the repository root in the environment described above. Each program writes new output separately from `retained_results/`.

### Complete suite

Import the geodesic distance files before starting the full suite.

```bash
cd code/complete
python run_all.py --stage plan --batch-size 32 --epochs 250 --out ../../new_runs/complete
python run_all.py --stage all --batch-size 32 --epochs 250 --device cuda --threads 4 --out ../../new_runs/complete
```

The `all` stage checks inputs and model execution before training. Individual stages are `core`, `fraction`, `tables`, `neighbors` and `threshold`; `priority` runs core and fraction jobs. The plan deduplicates reused configurations.

### Uniform preprocessing

```bash
cd code/uniform
python run_uniform.py --stage smoke --batch-size 32 --epochs 250 --device cuda --threads 4 --out ../../new_runs/uniform
python run_uniform.py --stage all --batch-size 32 --epochs 250 --device cuda --threads 4 --out ../../new_runs/uniform
```

Prepared masks are included. The program checks code and input files against `release_manifest.json` before execution.

### Component comparisons

```bash
cd code/components
python run_gpu.py --stage smoke --batch-size 32
python run_gpu.py --stage main --batch-size 32 --epochs 250 --threads 4
python run_gpu.py --stage smoke --batch-size 25
python run_gpu.py --stage main --batch-size 25 --epochs 250 --threads 4
```

Output is written to `code/components/gpu_results/`. Specify the batch sizes as shown: the historical default is 4. The `main` stage gives the 72 reported comparisons. Its `all` stage additionally starts a separate fraction experiment that is not used in Figure 6. A partial run can be restarted with `--resume-partial`; optimizer state is not resumed mid-run.

## Files

- `code/`: training programs, prepared inputs, manifests and fixed data splits.
- `retained_results/`: configurations, training histories and predictions for the reported runs.
- `analysis/`: statistical checks and the figure/table mapping.
- `tools/`: evaluation and figure-generation commands.
- `figure_assets/`: method diagrams and original scan views.
- `environment/`: dated server and package records.
- `recalculated_statistics/`: completed statistics check.

File hashes are recorded in `staging_manifest.json`, `retained_results/file_manifest.json` and `RELEASE_FILE_SHA256.json`. The staging manifest records the initial preparation on 23 October; its availability notes predate checkpoint recovery. Current contents and verification are listed in `docs/RELEASE_STATUS.txt`. Metadata with private paths has been sanitized. The manifests retain the corresponding source hashes for matching historical inputs and checkpoints; newly trained runs receive their own identities.

## Citation and attribution

Wang, Y., Sun, W., Jin, J., Kong, Z., and Yue, X. (2023). **MVGCN: Multi-View Graph Convolutional Neural Network for Surface Defect Identification Using Three-Dimensional Point Cloud.** *Journal of Manufacturing Science and Engineering*, 145(3), 031004. [Repository and paper link](https://github.com/wyn430/MVGCN).

Wu, K., Wang, X., and Yue, X. **Multi-view Dynamic Graph Deep Learning for Surface Defect Identification from 3D Point Cloud.** *IISE Transactions*. Publication details will be added when available.

Please cite both works when using these materials. Third-party code attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The original datasets retain their source attribution and terms.
