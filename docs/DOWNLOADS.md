# Archives and extraction

The archives below are prepared for the MDGCN release. [Wukexin1/MDGCN](https://github.com/Wukexin1/MDGCN) is private during preparation, with a public release planned. Download links will be added after the uploads have been completed and checked.

| Archive | Contents |
|---|---|
| `MDGCN_source_review_20261025.zip` | Code, documentation, fixed split definitions, statistical expectations and verification records |
| `MDGCN_inputs_and_results_20261025.zip` | Prepared inputs for all three programs; configurations, histories and saved predictions for 471 runs |
| `MDGCN_checkpoints_001.zip` | 213 complete-suite checkpoints |
| `MDGCN_checkpoints_002.zip` | The other 150 complete-suite checkpoints |
| `MDGCN_checkpoints_003.zip` | 36 uniform-preprocessing checkpoints |
| `MDGCN_checkpoints_004.zip` | 72 component checkpoints: 36 at batch size 25 and 36 at batch size 32 |
| `MDGCN_geodesic_001.zip` through `MDGCN_geodesic_004.zip` | All 320 geodesic distance matrices |

Extract the source archive into a new directory. Extract the other archives into that same directory, where `README.md` and `code/` reside. Each ZIP is independent; do not concatenate them. The source files, prepared inputs, checkpoint files and distance matrices have separate paths, so they can be extracted in any order after the source archive.

For example, from the directory containing the downloaded archives:

```bash
python -m zipfile -e MDGCN_source_review_20261025.zip MDGCN
python -m zipfile -e MDGCN_inputs_and_results_20261025.zip MDGCN
python -m zipfile -e MDGCN_checkpoints_001.zip MDGCN
python -m zipfile -e MDGCN_checkpoints_002.zip MDGCN
python -m zipfile -e MDGCN_checkpoints_003.zip MDGCN
python -m zipfile -e MDGCN_checkpoints_004.zip MDGCN
```

For geodesic experiments, also extract each of the four `MDGCN_geodesic_*.zip` files with the same command and destination. They are not needed for Euclidean inference or for recalculating statistics from saved predictions.

The resulting layout is:

```text
MDGCN/
  README.md
  code/complete/data/
  code/uniform/data/
  code/components/data/
  retained_results/
  recovered_checkpoints/complete/runs/<run_id>/selected.pt
  recovered_checkpoints/uniform/runs/<run_id>/selected.pt
  recovered_checkpoints/components/runs/<run_id>/selected.pt
  release_manifests/
```

Geodesic matrices are placed directly under `code/complete/data/surface/geo/` and `code/complete/data/real/geo/`. The four checkpoint archives contain all 471 selected models. Their saved predictions are in `retained_results/` for statistical verification.

`SHA256SUMS.txt` records the checksum of each archive. The per-file checksums in `release_manifests/`, the dataset manifests and the evaluation tool identify the exact inputs and model files.
