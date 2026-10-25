# Data versions, labels and layout

## Source and release scope

The 200 synthetic and 120 real point clouds originate from the MVGCN materials shared by their authors:

- [MVGCN source repository](https://github.com/wyn430/MVGCN).
- [Synthetic data folder](https://drive.google.com/drive/folders/11d5L1DJomhu_RCGtVBslECfsJv48o7AC).
- [Real data folder](https://drive.google.com/drive/folders/1qQh3wgQQpUl0genZ-yklOI1F4FcpuhMS).

The data provider has permitted sharing the additional preprocessing outputs and splits prepared for this study. Original data are referenced through the provider's repository and folders. The exact experimental inputs and derived files are distributed separately at `[VERSIONED_DATA_URL_PENDING]`. Some local synthetic inputs differ from a currently linked basic synthetic variant; the frozen input arrays and their SHA-256 records identify the files used for this paper.

Each original point cloud contains 2,000 points. File-level splits are fixed:

| Dataset | Categories | Training | Validation | Test | Total |
|---|---|---:|---:|---:|---:|
| `surface` | Crack, Dent, Normal, Raised | 128 | 32 | 40 | 200 |
| `real` | Dent, Normal, Uneven | 57 | 15 | 48 | 120 |

Class indices follow each manifest's `class_names` order. The same split is used for all three seeds. For real point clouds, the exact crop locations and overlap relationships within the scanned concrete specimen are not recorded; the released split describes point cloud files, not independent physical specimens.

## Three types of point labels

**Supplied region partitions.** In abnormal point clouds, the supplied binary values come from surface-variation thresholding: 1 denotes a candidate abnormal region and 0 a candidate normal region. In normal point clouds, the binary values are grouping labels. These files are separate from the point cloud category labels used for classification.

**Recomputed candidate masks.** The uniform and matched-component protocols derive masks using their recorded preprocessing rules. For the uniform protocol, the surface-variation statistic uses the center point and 20 neighbors; its threshold is the pooled normal-training mean plus 1.282 times the population standard deviation. The same dataset-specific threshold is then applied to every point cloud. The component caches retain their own calibration and masks. See the respective manifests; do not exchange masks between these protocols.

**Controlled support masks.** The generated controlled-fraction point clouds retain `gt_mask` and displacement fields describing the synthetic injected support. The network input readers explicitly load coordinates and candidate masks and do not load `gt_mask` or displacement as features.

## Frozen experimental inputs

| Path | Contents |
|---|---|
| `code/complete/data/surface/inputs.npz` | Synthetic XYZ, normals, labels and supplied/percentile/geodesic mask variants. |
| `code/complete/data/real/inputs.npz` | Real XYZ, normals, labels and supplied/percentile/geodesic mask variants. |
| `code/complete/data/{surface,real}/geo/` | Separate per-point-cloud geodesic matrices; populate from the archive below. |
| `code/uniform/data/{surface,real}/inputs.npz` | The corresponding XYZ, normals and labels, with `surface_variation` and `mask_uniform`. |
| `code/components/data/original_synthetic_v1/` | Per-point-cloud NPZ files and the matched-component manifest. |
| `code/components/data/original_industrial_v1/` | Per-point-cloud NPZ files and the matched-component manifest. |
| `code/complete/data/controlled/` | Controlled-fraction inputs used for Figure 6 and Table S1. |
| `code/components/data/controlled_v1/` | A physically separate copy of the controlled inputs, used by the component smoke test. |

Complete/uniform cached XYZ is centered and divided by the maximum radius; the supplied normals are retained. Component loaders center and scale coordinates as specified in their code. Each `manifest.json` lists sample IDs, classes, splits and preprocessing metadata. `inputs_sha256`, per-file hashes and the staging manifest identify the arrays.

The controlled cache has 550 files derived from 50 normal source point clouds: one unchanged normal sample per source and dent/raised samples at support fractions 0.005, 0.01, 0.02, 0.05 and 0.10. Sources are split before generating variants. The complete cache contains 341 training, 88 validation and 121 test files across all fractions; a run loads the normal samples and the affected samples at its chosen fraction. Generated variants of the same source do not cross the recorded partitions.

## Geodesic cache

The cache is distributed in four independent archives, `MDGCN_geodesic_001.zip` through `MDGCN_geodesic_004.zip`, at `[GEODESIC_CACHE_URL_PENDING]`. Together they contain 320 float32 matrices, each 2,000 × 2,000, totaling `5120040960` bytes of NPY files when unpacked. Extract all four archives into the repository root; they place the files directly at:

```text
code/complete/data/surface/geo/
code/complete/data/real/geo/
```

All 320 matrix files were checked against the dataset manifests after packaging. The four parts retain the matrix bytes from the earlier local `IISE_geodesic_cache_v1.zip` archive (SHA-256 `dfeb61a6f4fb123325e3e7d2cf57187dbbf0abb0ec965d382462ae8263ac7482`). Archive checksums and extraction commands are supplied with the release; see [DOWNLOADS.md](DOWNLOADS.md).

Keep the filenames unchanged. The complete runner validates every matrix against the dataset manifest. Geodesic variants use their paired coordinates, normals and masks from the frozen cache. The Euclidean arrays must not be substituted into that input variant. The uniform and component formal runs do not require geodesic matrices.

## Checkpoints and retained results

Retained predictions, configurations, histories and selected checkpoints cover all 471 runs: 363 complete, 36 uniform and 72 components. Every checkpoint passed an independent check against its historical SHA-256 value. Extract all four checkpoint archives into the repository root. The paths below are relative to `recovered_checkpoints/`:

```text
complete/runs/<run_id>/selected.pt
uniform/runs/<run_id>/selected.pt
components/runs/<run_id>/selected.pt
```

Use `tools/evaluate_saved.py` to compare a recovered file's SHA-256 with the recorded selected-checkpoint hash. A matching filename alone does not identify a matching model. The verified checkpoint assets will be linked at `[CHECKPOINT_URL_PENDING]` when published.

## Material excluded from public release

The full `Front_fusion_30.ply` scan is retained privately and is not needed for the patch classification experiments. Private correspondence, personal machine paths, credentials, author-administration materials and unselected historical runs are also excluded. Figure 5 uses the retained scan-view images; it is not presented as a new rendering of the released patch inputs. Raw scan registration and historical crop-camera reconstruction are outside the released experiment workflow.

Source and candidate hashes may differ for metadata whose private path fields were sanitized. This is recorded in `staging_manifest.json`; numerical arrays and historical result identity remain separately traceable. Some metadata refers to relative `raw_data/...` provenance identifiers; these are not files included in this candidate.
