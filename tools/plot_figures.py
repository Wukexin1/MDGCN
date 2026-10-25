#!/usr/bin/env python3
"""Reproduce manuscript Figures 4 and 6 from frozen data and saved predictions.

Python 3.8+, NumPy and Matplotlib are required. No checkpoint, GPU, training,
network connection or manuscript source is used. Run from any directory.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time


SURFACE_INPUT_SHA256 = "42d30d64f8408440471d4bdb77eca704ca746d8bb40e6566b0d6db3cfb24d6c8"
EXAMPLES = (
    ("Raised", "raised/raised1.txt", 412),
    ("Dent", "dent/dent1.txt", 193),
    ("Crack", "crack/crack1.txt", 578),
    ("Normal", "normal/normal10.txt", 113),
)
FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1)
SERIES = (
    ("dsrf", "DSRF-only", "#60666b", "s", -0.08),
    ("hard", "MDGCN1 (hard)", "#17856a", "o", 0.0),
    ("soft", "MDGCN2 (soft)", "#4664b0", "^", 0.08),
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_statistics_module(root):
    path = root / "analysis" / "reproduce_statistics.py"
    spec = importlib.util.spec_from_file_location("iise_statistics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def figure4(root, output, np, plt, dpi):
    source = root / "code" / "uniform" / "data" / "surface" / "inputs.npz"
    manifest_path = source.with_name("manifest.json")
    if digest(source) != SURFACE_INPUT_SHA256:
        raise ValueError("Figure 4 requires the frozen surface inputs with the published SHA-256.")
    manifest = load_json(manifest_path)
    audit = {"figure": 4, "source": "code/uniform/data/surface/inputs.npz",
             "input_sha256": digest(source), "manifest_sha256": digest(manifest_path),
             "point_mask": "mask_uniform", "raster_layer_dpi": dpi,
             "camera_elevation": 25, "camera_azimuth": -60, "examples": []}
    style = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with np.load(source, allow_pickle=False) as data, plt.rc_context(style):
        if data["xyz"].shape != (200, 2000, 3) or data["mask_uniform"].dtype != np.bool_:
            raise ValueError("Unexpected frozen point-cloud dimensions or mask type.")
        fig = plt.figure(figsize=(7, 5.7))
        for panel, (category, identifier, expected_candidates) in enumerate(EXAMPLES):
            indices = [i for i, record in enumerate(manifest["records"])
                       if record["id"] == identifier and record["category"] == category
                       and record["split"] == "train"]
            if len(indices) != 1:
                raise ValueError("The Figure 4 example is missing or duplicated: " + identifier)
            i = indices[0]
            xyz, mask = data["xyz"][i], data["mask_uniform"][i]
            if not np.isfinite(xyz).all() or int(mask.sum()) != expected_candidates:
                raise ValueError("Figure 4 coordinates or candidate mask differ from the frozen input.")
            ax = fig.add_subplot(2, 2, panel + 1, projection="3d")
            ax.scatter(*xyz[~mask].T, s=1.8, c="#607d8b", alpha=0.55, rasterized=True)
            ax.scatter(*xyz[mask].T, s=3, c="#c44e32", alpha=0.9, rasterized=True)
            ax.set_title(category, pad=0)
            ax.set_xlabel("x", labelpad=-4)
            ax.set_ylabel("y", labelpad=-4)
            ax.set_zlabel("z", labelpad=-4)
            ax.tick_params(labelsize=8, pad=-2)
            ax.view_init(25, -60)
            audit["examples"].append({"panel": panel + 1, "category": category,
                "sample_id": identifier, "row_index": i, "split": "train",
                "point_count": int(len(xyz)), "candidate_count": int(mask.sum()),
                "background_count": int((~mask).sum()),
                "coordinate_array_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
                "mask_array_sha256": hashlib.sha256(mask.tobytes()).hexdigest()})
        fig.subplots_adjust(left=0.01, right=0.95, bottom=0.03, top=0.94,
                            wspace=0.06, hspace=0.05)
        fig.savefig(output / "synthetic_examples.pdf", dpi=dpi)
        fig.savefig(output / "synthetic_examples.png", dpi=dpi)
        plt.close(fig)
    save_json(output / "figure4_data.json", audit)
    return audit


def controlled_statistics(root, np):
    calculator = load_statistics_module(root)
    expected = load_json(root / "analysis" / "expected_display_values.json")
    expected_runs = {r["key"]: r for r in expected["runs"]}
    run_root = calculator.result_directory(root / "retained_results" / "complete_return", "complete")
    checks = {c["group_id"]: c for c in expected["performance_checks"]
              if c["item"] == "Figure 6" and c["metric"] == "test_oa"}
    expected_config = {"family": "complete", "dataset": "controlled", "variant": "xyz",
                       "k": 15, "tau": "base", "batch_size": 32, "epochs": 250}
    groups, run_audit = {}, []
    for model, _, _, _, _ in SERIES:
        for fraction in FRACTIONS:
            matches = [g for g in expected["groups"]
                       if all(g["configuration"].get(k) == v for k, v in expected_config.items())
                       and g["configuration"].get("model") == model
                       and g["configuration"].get("rho") == fraction]
            if len(matches) != 1:
                raise ValueError("A required Figure 6 group is missing or duplicated.")
            group = matches[0]
            if group["seeds"] != [24, 42, 2026]:
                raise ValueError("Figure 6 requires seeds 24, 42 and 2026.")
            runs = {}
            for key in group["run_keys"]:
                run = expected_runs[key]
                if Path(run["run_id"]).name != run["run_id"]:
                    raise ValueError("Invalid retained run identifier.")
                verified = calculator.verify_run(run_root / run["run_id"], run, group)
                runs[key] = verified
                run_audit.append({k: verified[k] for k in (
                    "key", "seed", "selected_epoch", "epochs_recorded", "retained_file_sha256")})
            aggregate = calculator.aggregate_group(group, runs)
            values = aggregate["statistics"]["test_oa"]
            check = checks[group["group_id"]]
            if not (calculator.display_equal(values["mean_percent"], check["expected_mean"], check["decimals"])
                    and calculator.display_equal(values["sample_sd_percent"], check["expected_sample_sd"], check["decimals"])):
                raise ValueError("Recomputed Figure 6 statistics differ from the manuscript values.")
            groups[(model, fraction)] = {"group_id": group["group_id"], "seeds": group["seeds"],
                "run_keys": group["run_keys"], "mean_percent": values["mean_percent"],
                "sample_sd_percent": values["sample_sd_percent"],
                "seed_values_percent": values["seed_values_percent"]}
    return groups, run_audit


def figure6(root, output, np, plt, dpi):
    groups, runs = controlled_statistics(root, np)
    audit = {"figure": 6, "metric": "test point-cloud overall accuracy (%)",
             "standard_deviation": "sample standard deviation across three seeds, ddof=1",
             "seeds": [24, 42, 2026], "verified_run_count": len(runs), "series": [], "runs": runs}
    # The accepted Figure 6 uses Matplotlib's default font size and frame.
    # Black text is explicit; curve colours and symbols retain their original meaning.
    with plt.rc_context({"text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black"}):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for model, label, colour, marker, offset in SERIES:
            values = [groups[(model, fraction)] for fraction in FRACTIONS]
            mean = [v["mean_percent"] for v in values]
            sd = [v["sample_sd_percent"] for v in values]
            ax.errorbar(np.arange(5) + offset, mean, yerr=sd, color=colour,
                        marker=marker, capsize=3, label=label)
            audit["series"].append({"model": model, "label": label,
                "fractions": list(FRACTIONS), "means": mean, "sample_sds": sd,
                "colour": colour, "marker": marker, "groups": values})
        ax.set_xticks(range(5))
        ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
        ax.set_xlabel("Fraction of defect points (%)")
        ax.set_ylabel("Test OA (%)")
        ax.set_ylim(30, 100)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=3, loc="upper center", fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(output / "fraction_classification.pdf")
        fig.savefig(output / "fraction_classification.png", dpi=dpi)
        plt.close(fig)
    save_json(output / "figure6_data.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root (default: parent of tools/).")
    parser.add_argument("--output", type=Path,
                        help="Output folder (default: ROOT/generated/figures).")
    parser.add_argument("--figure", choices=("all", "4", "6"), default="all")
    parser.add_argument("--dpi", type=int, default=600,
                        help="PNG and Figure 4 point-layer resolution; default 600.")
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("Use at least 150 dpi (600 is the publication default).")
    root = args.root.resolve()
    output = (args.output if args.output is not None else root / "generated" / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep Matplotlib's cache in the requested output folder, not a user's home directory.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    started = time.perf_counter()
    audit = {"python": platform.python_version(), "numpy": np.__version__,
             "matplotlib": matplotlib.__version__, "training_performed": False,
             "inference_performed": False, "figures": []}
    if args.figure in ("all", "4"):
        audit["figures"].append(figure4(root, output, np, plt, args.dpi))
    if args.figure in ("all", "6"):
        audit["figures"].append(figure6(root, output, np, plt, args.dpi))
    audit["elapsed_seconds"] = time.perf_counter() - started
    audit["output_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                             for p in sorted(output.iterdir()) if p.suffix in (".pdf", ".png")}
    save_json(output / "figure_reproduction_audit.json", audit)
    print("Verified and generated Figure(s) {} in {}".format(
        ", ".join(str(f["figure"]) for f in audit["figures"]), output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
