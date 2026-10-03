# Replication Laboratory #2 — Can the Model Keep Learning?

**ANLY 735 — Research Seminar in Predictive AI**

## Laboratory Question

> What happens to learning behavior when conditions change?

This repository contains a focused **proxy replication** of the Week 4 claim
that a model may retain prior capability while becoming less effective at
learning what comes next under nonstationary conditions.

The experiment uses a reproducible synthetic classification task with a sudden
change in the labeling rule:

- **Task A:** classify points using the rule `x1 + x2 > 0`.
- **Task B:** classify points using the rule `x1 - x2 > 0`.

A previously trained, stability-biased agent first learns Task A and is then
updated on Task B. A fresh model starts directly on Task B. The comparison
uses the Stability-Plasticity Diagnostic:

```text
CHANGE -> RETENTION -> NEW LEARNING -> RATE -> TRADEOFF
```

## Repository Structure

```text
.
├── README.md
├── replication-lab.qmd
├── references.bib
├── analysis/
│   ├── README.md
│   ├── lab02_learning_trajectory.csv
│   └── lab02_summary.csv
├── figures/
│   └── lab02_learning_curves.svg
└── python/
    └── lab02_analysis.py
```

## Run the Analysis

No external Python packages are required. The analysis uses only the Python
standard library.

From the repository root, run:

```bash
python python/lab02_analysis.py
```

The script saves:

- `analysis/lab02_learning_trajectory.csv`
- `analysis/lab02_summary.csv`
- `figures/lab02_learning_curves.svg`

## Render the Report

Render the Quarto report to Word:

```bash
quarto render replication-lab.qmd --to docx
```

Submit the rendered `replication-lab.docx` file to Canvas. The GitHub
repository serves as the reproducibility record.
