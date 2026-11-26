"""Compare pruned teachers and distilled students across evaluation runs."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = {
    "svd": "SVD",
    "local_mag": "Local Magnitude",
    "global_mag": "Global Magnitude",
}
METRICS = (("Accuracy", "Accuracy"), ("F1Macro", "Macro F1"))
STUDENT_WITHOUT_KD = {"Accuracy": 77, "Macro F1": 0.54}
FILENAME_PATTERN = re.compile(r"evaluation_results_T(?P<T>[^_]+)_alpha(?P<alpha>[^.]+)\.csv$")


def _read_run(csv_path):
    """Read teacher and student metric values indexed by method and sparsity."""
    match = FILENAME_PATTERN.match(csv_path.name)
    if not match:
        raise ValueError(
            f"{csv_path.name} must match evaluation_results_T{{T}}_alpha{{alpha}}.csv"
        )
    alpha = float(match.group("alpha"))
    alpha_label = f"{alpha:.10f}".rstrip("0").rstrip(".")
    run_label = f"T={match.group('T')}, alpha={alpha_label}"
    data = defaultdict(dict)
    identity_rows = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Model", "Sparsity", "Accuracy", "F1Macro"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path}: expected columns {', '.join(sorted(required))}")
        for line, row in enumerate(reader, start=2):
            model = row["Model"].strip()
            if model in ("Pruned Teacher (identity)", "Distilled Student (identity)"):
                try:
                    values = {metric: float(row[column]) for column, metric in METRICS}
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{csv_path}:{line}: invalid identity metric") from exc
                identity_rows[model].append(values)
                continue
            for method in METHODS:
                teacher = f"Pruned Teacher ({method})"
                student = f"Distilled Student ({method})"
                if model not in (teacher, student):
                    continue
                try:
                    sparsity = float(row["Sparsity"])
                    values = {metric: float(row[column]) for column, metric in METRICS}
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{csv_path}:{line}: invalid sparsity or metric") from exc
                if model == teacher:
                    data[(method, "teacher")][sparsity] = values
                else:
                    data[(method, run_label)][sparsity] = values
    for model, rows in identity_rows.items():
        data[("identity", model)] = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for _, metric in METRICS
        }
    return run_label, data


def plot_multiple(csv_files=None, output_dir=None):
    """Create accuracy and macro-F1 plots for each pruning method.

    Each plot contains one pruned-teacher curve and one distilled-student
    curve per input CSV. Teacher values with the same method and sparsity are
    averaged across the runs; student values retain their run-specific labels.
    Returns the paths of the six generated PNG files.
    """
    script_dir = Path(__file__).resolve().parent
    files = [Path(path) for path in csv_files] if csv_files is not None else sorted(
        script_dir.glob("evaluation_results_T*_alpha*.csv")
    )
    if len(files) != 3:
        raise ValueError(f"Expected exactly 3 evaluation CSV files, found {len(files)}")
    if len({path.resolve() for path in files}) != 3:
        raise ValueError("The three evaluation CSV files must be distinct")

    runs = [_read_run(path) for path in files]
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Each CSV must have a distinct T and alpha filename")
    destination = Path(output_dir) if output_dir is not None else script_dir / "plots_multiple"
    destination.mkdir(parents=True, exist_ok=True)

    # Identity rows may repeat in the CSV (once per pruning experiment); first
    # average within each file, then give each of the three configurations equal weight.
    unpruned = {}
    for model, label in (("Pruned Teacher (identity)", "Unpruned teacher"),
                         ("Distilled Student (identity)", "Unpruned student")):
        per_run = [run_data[("identity", model)] for _, run_data in runs
                   if ("identity", model) in run_data]
        if not per_run:
            raise ValueError(f"No {label.lower()} identity rows found")
        unpruned[label] = {
            metric: sum(values[metric] for values in per_run) / len(per_run)
            for _, metric in METRICS
        }

    outputs = []
    colors = {"teacher": "black", **dict(zip(labels, ("tab:blue", "tab:orange", "tab:green")))}
    for method, display_name in METHODS.items():
        teacher_values = defaultdict(list)
        for _, run_data in runs:
            for sparsity, values in run_data.get((method, "teacher"), {}).items():
                teacher_values[sparsity].append(values)
        if not teacher_values:
            raise ValueError(f"No pruned teacher values found for {method}")
        teacher_map = {
            sparsity: {
                metric_name: sum(values[metric_name] for values in repeated_values) / len(repeated_values)
                for _, metric_name in METRICS
            }
            for sparsity, repeated_values in teacher_values.items()
        }
        teacher_sparsities = sorted(teacher_map)
        for column, metric_name in METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                teacher_sparsities,
                [teacher_map[s][metric_name] for s in teacher_sparsities],
                color=colors["teacher"], marker="s", linewidth=2,
                label="Pruned teacher",
            )
            ax.axhline(
                unpruned["Unpruned teacher"][metric_name], color="black",
                linestyle="--", linewidth=1.5, alpha=0.35,
                label="Unpruned teacher",
            )
            ax.axhline(
                unpruned["Unpruned student"][metric_name], color="gray",
                linestyle=":", linewidth=1.8, alpha=0.4,
                label="Unpruned student (mean)",
            )
            ax.axhline(
                STUDENT_WITHOUT_KD[metric_name], color="gray",
                linestyle="-.", linewidth=1.8, alpha=0.65,
                label="Student without KD",
            )
            for label, run_data in runs:
                student_map = run_data.get((method, label), {})
                points = sorted(
                    (sparsity, values[metric_name])
                    for sparsity, values in student_map.items()
                )
                if not points:
                    raise ValueError(f"No distilled-student values for {method} in {label}")
                x_values, y_values = zip(*points)
                ax.plot(x_values, y_values, marker="o", label=label, color=colors[label])
            ax.set_title(f"{metric_name}: {display_name} Pruning")
            ax.set_xlabel("Pruned teacher sparsity (%)")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.legend(title="Model")
            fig.tight_layout()
            output_path = destination / f"{method}_{column.lower()}_multiple.png"
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(output_path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Compare three evaluation runs by pruning method.")
    parser.add_argument("csv_files", nargs="*", type=Path, help="Three evaluation CSVs; defaults to matching files beside this script")
    parser.add_argument("--output-dir", type=Path, help="Plot output directory (default: plots_multiple)")
    args = parser.parse_args()
    if args.csv_files and len(args.csv_files) != 3:
        parser.error("provide exactly three CSV files")
    for path in plot_multiple(args.csv_files or None, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
