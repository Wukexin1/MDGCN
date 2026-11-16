# Downloads and extraction

The [MDGCN reproducibility release](https://github.com/Wukexin1/MDGCN/releases/tag/iise-repro-20261116) contains:

| File | Contents |
|---|---|
| `MDGCN_source_20261116.zip` | Training code, documentation, analysis scripts and verification records |
| [`MDGCN_inputs_and_results_20261025.zip`](https://github.com/Wukexin1/MDGCN/releases/download/iise-repro-20261116/MDGCN_inputs_and_results_20261025.zip) | Prepared inputs, fixed splits, configurations, training histories and saved predictions for 471 runs |
| `SHA256SUMS.txt` | SHA-256 checksums for the two archives |

If the code has already been cloned or downloaded, extract the inputs archive into its root, where `README.md` and `code/` reside:

```bash
python -m zipfile -e MDGCN_inputs_and_results_20261025.zip /path/to/MDGCN
```

To start from the source archive instead:

```bash
python -m zipfile -e MDGCN_source_20261116.zip MDGCN
python -m zipfile -e MDGCN_inputs_and_results_20261025.zip MDGCN
```

Each archive is independent. The result contains `code/complete/data/`, `code/uniform/data/`, `code/components/data/`, and `retained_results/`. The analysis and figure-generation commands in README.md use these files directly.

## Geodesic distance files

Download the original distance files through the MVGCN authors' [synthetic data folder](https://drive.google.com/drive/folders/11d5L1DJomhu_RCGtVBslECfsJv48o7AC) and [real data folder](https://drive.google.com/drive/folders/1qQh3wgQQpUl0genZ-yklOI1F4FcpuhMS). The synthetic geodesic folder is `Surface_Defects_pcd_extend_2000_geod_estnorm`; the real folder is `real_data_new_2000_geod`.

The importer uses the historical local folder name ending in `_noise0001` for the synthetic data. Place or rename the downloaded synthetic geodesic folder accordingly, retaining its class subdirectories and filenames:

```text
Data/
  Surface_Defects_pcd_extend_2000_geod_estnorm_noise0001/
    crack/  dent/  normal/  raised/
  real_data_new_2000_geod/
    dent/  normal/  uneven/
```

Each sample requires its `.txt` point file and paired `_geodesic_dis.npz` distance file. From `code/complete`, run:

```bash
python -c "from prepare import import_geo; import_geo('/absolute/path/to/Data', 'data')"
```

This imports the supplied distances into `data/surface/geo/` and `data/real/geo/`. It verifies the original point file, the original distance file and the converted matrix against the recorded SHA-256 values. It leaves the prepared inputs and data splits unchanged. If a file hash differs, the import stops; the recorded inputs should not be overwritten with another preprocessing variant.

These matrices are needed for the complete suite's geodesic experiments. Euclidean experiments, the uniform and component protocols, saved-result statistics and Figures 4 and 6 do not require them.

## Model checkpoints

Historical model checkpoints are not included in this release. The training programs generate checkpoints using the supplied inputs, splits, seeds and settings. The retained predictions allow the reported statistics to be recalculated without retraining. `tools/evaluate_saved.py` can additionally check an existing historical checkpoint against its recorded hash.
