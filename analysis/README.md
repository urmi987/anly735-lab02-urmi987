# Analysis Directory

This directory stores generated evidence for **Replication Laboratory #2 —
Can the Model Keep Learning?**

Running the analysis script creates:

```text
lab02_learning_trajectory.csv
lab02_summary.csv
```

`lab02_learning_trajectory.csv` records Task A and Task B accuracy after each
Task B training batch for both agents.

`lab02_summary.csv` records the compact stability-plasticity evidence:

- initial Task A accuracy;
- initial Task B accuracy;
- final Task A accuracy;
- final Task B accuracy;
- Task A retention change;
- Task B learning gain; and
- batches required to reach 80 percent Task B accuracy.

The script produces evidence. The interpretation belongs in
`replication-lab.qmd`.
