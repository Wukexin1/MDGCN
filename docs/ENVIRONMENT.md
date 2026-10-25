# Environment

## Recorded training environment

The training records specify Python 3.8.20, PyTorch 2.4.1, CUDA 11.8, an NVIDIA A100 GPU with 80 GB memory, and four CPU threads per training process. The component smoke record also specifies NumPy 1.24.3. The recorded Linux kernel was 5.15.0-187-generic.

## Server snapshot collected on 25 October 2026

Two collection runs returned the same software and hardware information. Only their collection times differ. The full version record is in [current_server_environment_20261025.json](../environment/current_server_environment_20261025.json); the [package list](../environment/current_server_package_versions_20261025.txt) contains installed names and versions.

| Item | Current server value |
|---|---|
| Operating system | Ubuntu 22.04.5 LTS, x86_64 |
| Linux kernel | 5.15.0-198-generic |
| Python | 3.8.20 |
| PyTorch / CUDA runtime | 2.4.1 / 11.8 |
| cuDNN | 9.1.0 (PyTorch value: 90100) |
| NVIDIA driver | 550.54.14 |
| GPUs visible during collection | Four NVIDIA A100 80GB PCIe GPUs |
| CPU | Intel Xeon Platinum 8358 at 2.60 GHz; 128 logical CPUs visible |
| System memory | 2,163,935,322,112 bytes, about 2,015 GiB |
| NumPy | 1.24.3 |
| Matplotlib | 3.7.5 |
| SciPy / scikit-learn | 1.10.1 / 1.3.2 |

The kernel has changed since training. These snapshots describe the current server; values absent from the original training records retain this collection date. Four visible GPUs and total system memory describe the server capacity, not the resources consumed by a single run.

## Install the required packages

The commands below create a separate Linux environment. Keep an existing training environment available for the first comparison against saved predictions.

```bash
conda create -n mdgcn_reproduce python=3.8.20 pip=24.2
conda activate mdgcn_reproduce
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-runtime.in
python -m pip install -r requirements-analysis.in
```

The PyTorch command follows the [official installation instructions for version 2.4.1](https://pytorch.org/get-started/previous-versions/). The requirements files specify the direct packages used by this release. Their versions come from the retained records and current server snapshot; they are not an export of all historical transitive dependencies. A complete reinstall with these commands still needs a fresh-environment smoke test.

| Operation | Direct packages |
|---|---|
| Recalculate statistics from saved predictions | NumPy |
| Plot Figures 4 and 6 | NumPy, Matplotlib |
| Train or evaluate using the frozen inputs | NumPy, PyTorch |
| Reconstruct neighborhoods or controlled point clouds | NumPy, SciPy |
| Reconstruct selected percentile masks | NumPy, scikit-learn |

For the optional data-reconstruction scripts, also install `requirements-preprocessing.in`. The released model definitions do not import torchvision, torchaudio, PyTorch Geometric or Open3D, and require no custom CUDA extension.

The server package inventory contains torchvision 0.20.0. The official pairing for PyTorch 2.4.1 is torchvision 0.19.1, so the full inventory is retained as a record rather than used as an installation file. The minimal environment above does not install torchvision or torchaudio.

Check the environment before running experiments:

```bash
python -c "import sys, numpy, torch; print(sys.version); print('NumPy', numpy.__version__); print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda); print('cuDNN', torch.backends.cudnn.version()); print('CUDA available', torch.cuda.is_available())"
python -m pip check
```

Then run the smoke stage for the intended experiment family using the commands in the README.

## Seeds and timing

The formal runs use seeds 24, 42 and 2026. All three training programs disable cuDNN benchmarking and enable deterministic settings. Complete and uniform use `torch.use_deterministic_algorithms(True, warn_only=True)`; components uses the same setting without warning-only mode. The evaluation report records numerical differences from the saved predictions together with the current device and package versions.

The retained logs cover 471 runs of 250 epochs each. Their recorded intervals are summarized below.

| Suite | Runs | Sum of recorded intervals | Median interval per run |
|---|---:|---:|---:|
| Complete | 363 | 49.57 hours | 7.61 minutes |
| Uniform preprocessing | 36 | 6.55 hours | 6.80 minutes |
| Components | 72 | 21.94 hours | 16.09 minutes |

For complete and uniform, the totals sum the per-epoch `seconds` values in `history.json`, including training and validation but excluding checkpoint and log writes after each epoch timer. For components, the totals use each run's `elapsed_training_seconds` from `final_metrics.json` once; its history contains cumulative readings. These component intervals include training, validation, checkpoint saving, logging, and loading and hashing the selected checkpoint before final evaluation. All three suites exclude initialization and final split evaluation from these intervals. Summed job intervals do not measure the calendar time required to run the full study.

The full saved-result statistics check took 2.62 seconds on a local arm64 macOS CPU with Python 3.9.13 and NumPy 1.26.4, including input checks and writing the verification outputs. This operation excludes figure generation and model inference. Four full test-set inference checks took 4.94 seconds (synthetic PointNet), 62.46 seconds (synthetic MDGCN1), 86.45 seconds (real-data MDGCN2), and 82.61 seconds (real-data component model with soft aggregation and add fusion), including input loading, using four CPU threads with PyTorch 2.6.0; their environments and results are recorded in [inference_summary.json](../verification/inference_summary.json). These local timings describe the listed checks rather than a GPU speed comparison.
