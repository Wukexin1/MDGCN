RECALCULATING THE RETAINED MANUSCRIPT RESULTS

Scope
-----
reproduce_statistics.py verifies the statistics in the final manuscript and
supplementary tables from saved predictions, training histories and run
configurations. It does not train a model, load a checkpoint, perform inference,
or generate figures. The input directories are read only.

Requirements
------------
Python 3.8 or newer and NumPy:

    python -m pip install numpy

No PyTorch, CUDA, GPU, plotting package, or point-cloud array is required.

Files in this analysis directory
-------------------------------
reproduce_statistics.py        Portable verification entry point.
expected_display_values.json   Fixed run membership and final displayed values.
figure_table_mapping.json      Coverage and provenance of Figures 1-6,
                              Tables 1-10, and supplementary Tables S1-S7.
README_results_only.txt        This document.

Required retained inputs
------------------------
Supply the extracted result returns as three directories. The complete and
uniform returns contain a results directory. The component return contains a
gpu_results directory and must include both batch-size 25 and batch-size 32
runs. The verifier selects only the 471 runs listed in the expected manifest;
unrelated, incomplete, and earlier batch-size 4 runs are ignored.

    complete_return/
      results/<run_id>/
        result.json
        config.json
        history.json
        train_predictions.json
        val_predictions.json
        test_predictions.json

    uniform_return/
      results/<run_id>/       (the same six filenames)
      data/surface/manifest.json
      data/real/manifest.json

    component_return/
      gpu_results/<run_id>/
        final_metrics.json
        config.json
        history.json
        train_predictions.json
        val_predictions.json
        test_predictions.json

The three arguments can also point directly to results or gpu_results. When
--uniform points to results, its parent must contain data/surface/manifest.json
and data/real/manifest.json. Input .zip files must be extracted first.

The local candidate now includes these retained result directories. In its 72
component config copies, only the original machine path fields named data and
out were removed. All numerical results and training parameters are retained.
The other 3,554 retained JSON files preserve their original bytes. Original run
identities were checked before copying; the removed paths do not enter those
identities. See retained_results/file_manifest.json for relative paths and hashes.

The entry point has been run on the candidate copies: 471 runs, 157 groups and
75,630 prediction rows passed; all 630 displayed mean/SD pairs and 30 candidate
statistics matched. The report is recalculated_statistics/verification_report.json.
This is a local release candidate and has not been published by this workflow.

Usage
-----
Run from any directory; quote paths containing spaces:

    python analysis/reproduce_statistics.py \
      --complete retained_results/complete_return \
      --uniform retained_results/uniform_return \
      --components retained_results/component_return \
      --output recalculated_statistics

The default expected manifest is located next to reproduce_statistics.py.
--expected can select a different compatible manifest explicitly. The same
output filenames are replaced on a repeated run, so use a fresh --output
directory if previous reports need to be preserved.

Exit status is 0 when all required records and displayed values match, or 1
when verification fails. The JSON report and available recalculations are
written even if some retained runs are missing or inconsistent.

Calculations and checks
-----------------------
1. Check each configured dataset, model, input variant, graph neighborhood,
   threshold rule, batch size, epoch count, seed, and class order against the
   fixed run manifest.
2. Rebuild train/validation/test confusion matrices from retained truth and
   prediction labels. Check that probability argmax gives each saved prediction.
   Recalculate overall accuracy (OA), balanced accuracy (BA), and macro F1.
   BA is the mean recall over classes with true support. Macro F1 averages over
   all configured classes, with zero assigned when the F1 denominator is zero.
3. Compare the recalculated metrics with the saved per-run summary. Check that
   the history contains epochs 1 through 250 without gaps or duplicates.
4. Verify checkpoint selection from the validation history: maximum validation
   OA for complete/uniform runs, or maximum validation BA for component runs;
   lower ordinary cross entropy breaks ties, followed by the earliest exact tie.
5. For each of the 157 groups, average per-seed metrics over seeds 24, 42 and 2026
   and compute the sample standard deviation with ddof=1. All three seeds use
   the same fixed split. This SD is not an interval over independently sampled
   physical specimens or independently randomized data partitions.
6. Compare 630 mean/SD pairs with the final tables and Figure 6 data. Table
   entries are compared at their displayed two-decimal precision. Figure 6
   curve and error-bar values use full precision with absolute tolerance 1e-10.
7. Recalculate 30 displayed values in Tables 2 and 5 from retained individual
   point-cloud counts in the uniform manifests. Each cloud has 2,000 points.
   These include category counts, supplied and calibrated partition percentages,
   calibrated ranges, and pooled background/candidate count ratios. Normal
   samples' dummy labels are excluded from these abnormal-category statistics.

Generated outputs
-----------------
verification_report.json       Overall result, counts, issues and scope.
runs_recalculated.json         Per-run configuration, selected epoch, metrics,
                              confusion matrices and retained-file hashes.
run_metrics.csv               Per-run and split OA/BA/macro F1, on a 0-1 scale.
group_summary.json / .csv      All nine split/metric summaries per group;
                              means, sample SDs and per-seed values in percent.
display_value_checks.json     One row per manuscript performance mean/SD pair.
display_value_checks.csv      The same checks in tabular form.
candidate_proportions.json     Recalculated per-category partition statistics.
candidate_proportions.csv      The same statistics in tabular form.
candidate_display_checks.json Final Table 2/5 value comparisons.
candidate_display_checks.csv  The same comparisons in tabular form.

The JSON records do not include absolute input paths, full original machine
configurations, or individual sample identifiers. Content hashes identify the
exact retained files read during this execution. A sanitized configuration file
will have a different content hash but can still pass the protocol checks.

Mapping notes
-------------
Tables 3 and 6 reuse complete-return PointNet, PointNet++ and DGCNN runs as the
mask-free baselines, with uniform-return MVGCN, MDGCN1 and MDGCN2 results.
Tables 4 and 7 use complete-return runs with the supplied point partitions and
input/distance variants. Table 8 and Tables S2-S5 use the matched component
study, including batch-size 32 and 25 results. These protocol families have
different implementations of smoothing and candidate handling and are kept
separate in the group keys.

Table S1 and Figure 6 share the controlled-fraction groups. Tables 9/S6 share
neighborhood groups, and Tables 10/S7 share threshold-percentile groups. Matching
k=15 core runs are reused rather than counted as separate training trials.

Verification scope and separate evaluation/figure tools
------------------------------------------------------
Passing this script establishes agreement between saved predictions, run
records and final displayed statistics. It does not establish that training or
inference was rerun. This verifier reads saved predictions; the selected model
checkpoints are distributed separately. Training and inference use the
corresponding code, environments, inputs and evaluation commands.

The candidate-count checks verify saved manifest counts only. They do not
recompute neighborhood covariance matrices, surface variation or calibrated
masks. The supplied/calibrated binary partitions are threshold-based candidate
regions and are not independent manual pointwise ground truth.

Figures 1-3 are author-provided method diagrams. They require no training.
Figure 4 sample selection and the required cache fields are recorded in
figure_table_mapping.json. The separate tools/plot_figures.py entry point
renders Figures 4 and 6 and verifies Figure 6 against the retained runs.
Figure 5 uses retained original scan views; camera settings and exact
patch-to-scan crop mappings are unavailable. Figure 6 numerical curve and
error-bar values are verified here. See docs/figures.txt for rendering commands
and docs/saved_evaluation.txt for the separate checkpoint inference entry point.

README.md and docs/DATA.md record the current data-sharing plan. Public archive
links, remaining code provenance and the authors' code license are completed
separately from this numerical verification.
