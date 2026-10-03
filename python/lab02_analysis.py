"""
ANLY 735 — Replication Laboratory #2
Can the Model Keep Learning?

This script is a package-free proxy replication of the stability-plasticity
claim. It uses only the Python standard library so the analysis can be rerun
without installing external dependencies.
"""

from pathlib import Path
import csv
import math
import random


RANDOM_SEED = 735
N_TRAIN_A = 5000
N_TEST = 4000
N_BATCHES_B = 30
BATCH_SIZE_B = 160
LEARNING_RATE_A = 0.12
LEARNING_RATE_B_STABLE = 0.015
LEARNING_RATE_B_FRESH = 0.08
STABILITY_PENALTY = 0.80
RECOVERY_THRESHOLD = 0.80

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ROOT / "figures"

ANALYSIS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def sigmoid(z):
    z = max(min(z, 40), -40)
    return 1.0 / (1.0 + math.exp(-z))


def make_task_data(rng, n, task):
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)

        if task == "A":
            y = 1.0 if x1 + x2 > 0 else 0.0
        elif task == "B":
            y = 1.0 if x1 - x2 > 0 else 0.0
        else:
            raise ValueError("task must be 'A' or 'B'")

        rows.append(([1.0, x1, x2], y))
    return rows


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def gradient_step(weights, rows, learning_rate, anchor=None, stability_penalty=0.0):
    gradient = [0.0 for _ in weights]

    for x, y in rows:
        prediction = sigmoid(dot(weights, x))
        error = prediction - y
        for j in range(len(weights)):
            gradient[j] += error * x[j]

    n = len(rows)
    for j in range(len(weights)):
        gradient[j] /= n
        if anchor is not None and stability_penalty > 0:
            gradient[j] += stability_penalty * (weights[j] - anchor[j])
        weights[j] -= learning_rate * gradient[j]

    return weights


def accuracy(weights, rows):
    correct = 0
    for x, y in rows:
        prediction = 1.0 if sigmoid(dot(weights, x)) >= 0.5 else 0.0
        if prediction == y:
            correct += 1
    return correct / len(rows)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def batches_to_threshold(rows, agent, threshold):
    for row in rows:
        if row["agent"] == agent and float(row["task_b_accuracy"]) >= threshold:
            return row["batch"]
    return ""


def points_to_polyline(points, width, height, margin, x_max, y_min, y_max):
    scaled = []
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    for x, y in points:
        sx = margin + (x / x_max) * plot_width
        sy = height - margin - ((y - y_min) / (y_max - y_min)) * plot_height
        scaled.append(f"{sx:.1f},{sy:.1f}")

    return " ".join(scaled)


def save_svg_learning_curve(rows, path):
    width = 900
    height = 560
    margin = 70
    x_max = N_BATCHES_B
    y_min = 0.45
    y_max = 1.00

    colors = {
        "previously_trained_stability_biased": "#2f6f9f",
        "fresh_task_b_model": "#b44b33",
    }
    labels = {
        "previously_trained_stability_biased": "Previously trained, stability-biased",
        "fresh_task_b_model": "Fresh Task B model",
    }

    threshold_y = height - margin - (
        (RECOVERY_THRESHOLD - y_min) / (y_max - y_min)
    ) * (height - 2 * margin)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">',
        '<rect width="900" height="560" fill="white"/>',
        '<text x="450" y="35" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">Learning After the Condition Change</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{margin}" y1="{threshold_y:.1f}" x2="{width-margin}" y2="{threshold_y:.1f}" stroke="#777" stroke-dasharray="7 7" stroke-width="1.5"/>',
        f'<text x="{width-margin-4}" y="{threshold_y-8:.1f}" text-anchor="end" font-family="Arial" font-size="14" fill="#666">80% recovery threshold</text>',
        f'<text x="450" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="16">Task B training batch</text>',
        f'<text x="20" y="280" text-anchor="middle" transform="rotate(-90 20 280)" font-family="Arial" font-size="16">Task B accuracy</text>',
    ]

    for y_tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sy = height - margin - ((y_tick - y_min) / (y_max - y_min)) * (
            height - 2 * margin
        )
        lines.append(
            f'<line x1="{margin-5}" y1="{sy:.1f}" x2="{width-margin}" y2="{sy:.1f}" stroke="#ddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin-12}" y="{sy+5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{y_tick:.1f}</text>'
        )

    for x_tick in [0, 5, 10, 15, 20, 25, 30]:
        sx = margin + (x_tick / x_max) * (width - 2 * margin)
        lines.append(
            f'<line x1="{sx:.1f}" y1="{height-margin}" x2="{sx:.1f}" y2="{height-margin+5}" stroke="#222" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{sx:.1f}" y="{height-margin+24}" text-anchor="middle" font-family="Arial" font-size="13">{x_tick}</text>'
        )

    for agent in labels:
        points = [
            (int(row["batch"]), float(row["task_b_accuracy"]))
            for row in rows
            if row["agent"] == agent
        ]
        polyline = points_to_polyline(points, width, height, margin, x_max, y_min, y_max)
        lines.append(
            f'<polyline points="{polyline}" fill="none" stroke="{colors[agent]}" stroke-width="3"/>'
        )
        for x, y in points[::3]:
            sx = margin + (x / x_max) * (width - 2 * margin)
            sy = height - margin - ((y - y_min) / (y_max - y_min)) * (
                height - 2 * margin
            )
            lines.append(
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="{colors[agent]}"/>'
            )

    legend_x = 520
    legend_y = 84
    for i, agent in enumerate(labels):
        y = legend_y + i * 28
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+35}" y2="{y}" stroke="{colors[agent]}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x+45}" y="{y+5}" font-family="Arial" font-size="14">{labels[agent]}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rng = random.Random(RANDOM_SEED)

    train_a = make_task_data(rng, N_TRAIN_A, "A")
    test_a = make_task_data(rng, N_TEST, "A")
    test_b = make_task_data(rng, N_TEST, "B")

    stable_weights = [0.0, 0.0, 0.0]
    for _ in range(90):
        stable_weights = gradient_step(stable_weights, train_a, LEARNING_RATE_A)

    task_a_anchor = stable_weights[:]
    fresh_weights = [0.0, 0.0, 0.0]
    records = []

    for batch in range(N_BATCHES_B + 1):
        records.append(
            {
                "agent": "previously_trained_stability_biased",
                "batch": batch,
                "task_a_accuracy": f"{accuracy(stable_weights, test_a):.4f}",
                "task_b_accuracy": f"{accuracy(stable_weights, test_b):.4f}",
            }
        )
        records.append(
            {
                "agent": "fresh_task_b_model",
                "batch": batch,
                "task_a_accuracy": f"{accuracy(fresh_weights, test_a):.4f}",
                "task_b_accuracy": f"{accuracy(fresh_weights, test_b):.4f}",
            }
        )

        if batch == N_BATCHES_B:
            break

        batch_b = make_task_data(rng, BATCH_SIZE_B, "B")
        stable_weights = gradient_step(
            stable_weights,
            batch_b,
            LEARNING_RATE_B_STABLE,
            anchor=task_a_anchor,
            stability_penalty=STABILITY_PENALTY,
        )
        fresh_weights = gradient_step(fresh_weights, batch_b, LEARNING_RATE_B_FRESH)

    trajectory_path = ANALYSIS_DIR / "lab02_learning_trajectory.csv"
    write_csv(
        trajectory_path,
        records,
        ["agent", "batch", "task_a_accuracy", "task_b_accuracy"],
    )

    summary_rows = []
    for agent in ["previously_trained_stability_biased", "fresh_task_b_model"]:
        agent_rows = [row for row in records if row["agent"] == agent]
        first = agent_rows[0]
        final = agent_rows[-1]
        initial_a = float(first["task_a_accuracy"])
        initial_b = float(first["task_b_accuracy"])
        final_a = float(final["task_a_accuracy"])
        final_b = float(final["task_b_accuracy"])
        summary_rows.append(
            {
                "agent": agent,
                "initial_task_a_accuracy": f"{initial_a:.3f}",
                "initial_task_b_accuracy": f"{initial_b:.3f}",
                "final_task_a_accuracy": f"{final_a:.3f}",
                "final_task_b_accuracy": f"{final_b:.3f}",
                "task_a_retention_change": f"{final_a - initial_a:.3f}",
                "task_b_learning_gain": f"{final_b - initial_b:.3f}",
                "batches_to_80pct_task_b": batches_to_threshold(
                    records, agent, RECOVERY_THRESHOLD
                ),
            }
        )

    summary_path = ANALYSIS_DIR / "lab02_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "agent",
            "initial_task_a_accuracy",
            "initial_task_b_accuracy",
            "final_task_a_accuracy",
            "final_task_b_accuracy",
            "task_a_retention_change",
            "task_b_learning_gain",
            "batches_to_80pct_task_b",
        ],
    )

    figure_path = FIGURES_DIR / "lab02_learning_curves.svg"
    save_svg_learning_curve(records, figure_path)

    print("Saved:")
    print(f"- {trajectory_path}")
    print(f"- {summary_path}")
    print(f"- {figure_path}")
    print("\nSummary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
