# Data Directory

Replication Laboratory #2 uses synthetic data generated directly by:

```text
python/lab02_analysis.py
```

No external dataset is required. The script generates two-dimensional
classification observations with a fixed random seed.

- **Task A:** the label is determined by `x1 + x2 > 0`.
- **Task B:** the label is determined by `x1 - x2 > 0`.

The change from Task A to Task B is the experimental condition change used to
evaluate retention, new learning, learning rate, and tradeoff.
