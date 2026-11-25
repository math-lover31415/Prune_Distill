import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = (
    ("Accuracy", 2, 100.0), #column what to divide by
    ("Macro F1", 3, 1.0),
    ("Weighted F1", 4, 1.0),
    ("Macro Precision", 5, 1.0),
    ("Macro Recall", 6, 1.0),
)
METHODS = {
    "svd": "SVD",
    "local_mag": "Local Magnitude",
    "global_mag": "Global Magnitude",
}


def load_results(csv_path):
    results = defaultdict(lambda: defaultdict(list))
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        for line_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row:
                continue
            if len(row) < 7:
                raise ValueError(
                    f"{csv_path}:{line_number}: expected at least 7 columns, got {len(row)}"
                )
            try:
                severity = float(row[1])
                values = [float(row[column]) / scale for _, column, scale in METRICS]
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{line_number}: invalid numeric value in evaluation row"
                ) from exc
            results[row[0]][severity].append(values)
    if not results:
        raise ValueError(f"No evaluation rows found in {csv_path}")
    return results


def average_at_severities(results, labels): #because some run twice so some have 2 
    grouped = defaultdict(list)
    for label in labels:
        for severity, rows in results.get(label, {}).items():
            grouped[severity].extend(rows)
    return {
        severity: [
            sum(row[metric_index] for row in rows) / len(rows)
            for metric_index in range(len(METRICS))
        ]
        for severity, rows in grouped.items()
    }


def plot_method(results, method, display_name, output_dir):
    student_values = average_at_severities(
        results,
        ("Distilled Student (identity)", f"Distilled Student ({method})"),
    )
    teacher_values = average_at_severities(
        results,
        ("Pruned Teacher (identity)", f"Pruned Teacher ({method})"),
    )
    if not student_values or not teacher_values:
        raise ValueError(f"Missing student or teacher results for {method}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flat
    for metric_index, (metric_name, _, _) in enumerate(METRICS):
        ax = axes[metric_index]
        for series, label, color, marker in (
            (student_values, "Distilled student", "tab:blue", "o"),
            (teacher_values, "Pruned teacher", "tab:orange", "s"),
        ):
            severities = sorted(series)
            ax.plot(
                severities,
                [series[level][metric_index] for level in severities],
                label=label,
                color=color,
                marker=marker,
            )
        ax.set_title(metric_name)
        ax.set_xlabel("Pruning severity / sparsity (%)")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    max_severity = max(max(student_values), max(teacher_values))
    x_ticks = list(range(0, int(max_severity) + 1, 10))
    if max_severity not in x_ticks:
        x_ticks.append(max_severity)
    axes[0].set_xticks(x_ticks)
    axes[0].set_xlim(0, max_severity + max(1, max_severity * 0.02))

    axes[-1].set_visible(False)
    fig.suptitle(f"{display_name} Pruning: Student and Teacher Metrics")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_metrics.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics across pruning severity."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "evaluation_results.csv",
        help="Evaluation CSV (default: evaluation_results.csv beside this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory for generated plot images (default: plots beside this script)",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    for method, display_name in METHODS.items():
        output_path = plot_method(results, method, display_name, args.output_dir)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()