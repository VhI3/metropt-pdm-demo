# Cheat Sheet: Labeling Time Series Data for Supervised Learning

Derived from the MetroPT-3 air compressor predictive maintenance example.
Generalized for any domain: industrial IoT, healthcare monitoring, finance, IT operations, etc.

---

## The Core Problem

You have a long time series of measurements collected continuously over time. The raw data has **no event or label column** — it is just numbers over time. But you have external knowledge (from logs, reports, annotations, or domain experts) about *when* something important happened.

**The task:** Convert that external knowledge into a label column on the DataFrame so a machine learning model can learn what the signal looks like before, during, or after those events.

**Domain examples of the same pattern:**

| Domain | Raw signal | External knowledge | Label |
|---|---|---|---|
| Industrial IoT | Pressure, temperature, current | Maintenance report timestamps | Failure / normal |
| Healthcare | ECG, blood pressure | Doctor annotations | Arrhythmia episode / normal |
| Finance | Price, volume, volatility | News event dates | Market stress / normal |
| IT operations | CPU, memory, latency | Incident tickets | Outage / normal |
| Energy | Grid frequency, load | Blackout records | Fault / normal |

---

## Step 1 — Define Event Windows

```python
# General pattern — works for any domain
event_windows = [
    {"name": "event_A", "start": "YYYY-MM-DD HH:MM:SS", "end": "YYYY-MM-DD HH:MM:SS"},
    {"name": "event_B", "start": "YYYY-MM-DD HH:MM:SS", "end": "YYYY-MM-DD HH:MM:SS"},
]

# MetroPT-3 specific example
failure_windows = [
    {"name": "failure_1", "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00"},
    {"name": "failure_2", "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00"},
    {"name": "failure_3", "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00"},
    {"name": "failure_4", "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00"},
]
```

### What is this?
A list of dictionaries, each describing a known event with a name, a start timestamp, and an end timestamp. Each dictionary is one "ground truth" interval.

### Where do these intervals come from?
They come from **external sources, not from the data itself**:
- Maintenance logs or CMMS (Computerized Maintenance Management System) exports
- Incident tickets (e.g., PagerDuty, Jira, ServiceNow)
- Clinical annotations from doctors or nurses
- Official dataset documentation
- Domain expert interviews

### Why a list of dictionaries — not just a list of timestamps?
- The `name` field lets you trace which predictions correspond to which specific event later.
- Easily extended: adding a new event is one new line.
- Enables per-event analysis (e.g., some events last hours, others last days — behavior may differ).
- Can be loaded from a CSV or JSON file for larger projects (see extension section below).

### What if you do not have any labeled windows?
Then you **cannot do supervised learning**. Your options are:

| Approach | When to use | Limitation |
|---|---|---|
| Unsupervised anomaly detection | No labels at all | Cannot be objectively evaluated |
| Weak supervision / heuristics | Rules that approximate labels | Labels are noisy |
| Active learning | Domain expert labels a small sample | Requires expert time |
| Semi-supervised learning | A few labels + many unlabeled | More complex to implement |

Even a handful of labeled windows is far more valuable than none. If possible, prioritize getting at least a few confirmed ground-truth events.

---

## Step 2 — Initialize Default Labels

```python
df["label"]      = 0        # integer: 0 = normal, 1 = event
df["label_name"] = "normal" # string: human-readable class name
```

### Why initialize to 0 / "normal" first?
This is the **safe default** pattern — assume every row is normal until proven otherwise. If you skip this and only assign 1 to event rows, all other rows will have `NaN`, which breaks most ML models, aggregations, and plots.

### Why keep both a numeric and a string column?
- `label` (integer) → used by the model as the training target
- `label_name` (string) → used for debugging, plotting, and per-class analysis

The string column is free to add and saves significant debugging time later.

### Multi-class variant
If you have multiple distinct event types (not just normal vs. anomaly), use integer codes:

```python
df["label"] = 0  # 0 = normal
# Then per event type:
# 1 = air_leak, 2 = oil_leak, 3 = bearing_fault, etc.
```

Or use a categorical string column directly and encode it only at training time with `LabelEncoder`.

---

## Step 3 — Apply Labels Using a Boolean Mask

```python
for event in event_windows:
    start = pd.Timestamp(event["start"])
    end   = pd.Timestamp(event["end"])
    mask  = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df.loc[mask, "label"]      = 1
    df.loc[mask, "label_name"] = event["name"]
```

### What is a boolean mask?
A Series of `True`/`False` values, one per row, answering: "Does this row fall within the event window?" It is built by combining two comparisons with `&` (AND operator for element-wise logic on Series).

### Why `pd.Timestamp()` instead of raw strings?
Pandas cannot reliably compare plain strings to datetime columns. Converting to `pd.Timestamp` ensures type compatibility and correct comparisons. Always do this conversion explicitly.

### Why `>=` and `<=` (inclusive)?
Event reports give a start and end boundary. Inclusive comparisons ensure rows recorded exactly at those boundary timestamps are included. Using strict `>` / `<` silently drops the first and last readings of each event.

### Why `df.loc[mask]` instead of a row loop?
`df.loc` is **vectorized** — it runs in compiled C code across all matching rows at once. A Python `for` loop with `iterrows()` over a million-row DataFrame would be 100–1000x slower. Always use vectorized assignment.

### Handling overlapping windows
The loop applies events sequentially. If two windows overlap, the **last one processed wins** for the overlapping rows. Solutions:

```python
# Option 1: Sort windows by priority before looping (highest priority last)
event_windows = sorted(event_windows, key=lambda e: e["priority"])

# Option 2: Merge overlapping windows beforehand using pd.Interval
# Option 3: Use integer codes and assign the higher code last
```

### Adding a "pre-event" warning zone (common in predictive maintenance)
For early warning models, you may want to label a window *before* the event as a separate "degradation" class:

```python
warning_hours = 6  # how far in advance you want to predict

for event in event_windows:
    start = pd.Timestamp(event["start"])
    end   = pd.Timestamp(event["end"])

    # Label the failure itself
    mask_fail = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df.loc[mask_fail, "label"]      = 2
    df.loc[mask_fail, "label_name"] = event["name"]

    # Label the warning zone before the failure
    warn_start = start - pd.Timedelta(hours=warning_hours)
    mask_warn  = (df["timestamp"] >= warn_start) & (df["timestamp"] < start)
    df.loc[mask_warn, "label"]      = 1
    df.loc[mask_warn, "label_name"] = event["name"] + "_warning"
```

---

## Step 4 — Sanity Check

```python
print("Labeled rows:", df["label"].sum(), "/", len(df))
print(df["label"].value_counts())
```

### Why always run this check?
- If labeled count is **0**: timestamps did not match. Check for timezone mismatch, wrong column name, or wrong date format.
- If labeled count is **too high**: your windows are too wide, or the timestamp column has duplicates.
- `value_counts()` reveals the **class imbalance ratio** — critical for choosing the right model and metric.

### Diagnosing zero matches: a checklist

```python
# 1. Check the actual range of your timestamp column
print(df["timestamp"].min(), df["timestamp"].max())

# 2. Check for timezone info (tz-aware vs tz-naive mismatch)
print(df["timestamp"].dtype)
# If tz-aware: pd.Timestamp("2020-04-18", tz="UTC")

# 3. Check the column name is exactly right
print(df.columns.tolist())
```

### What is class imbalance and why does it matter?
Failures (or rare events) are a tiny fraction of the data. A model that always predicts "normal" will achieve 99%+ accuracy while being completely useless. Always use:
- **F1-score** (harmonic mean of precision and recall)
- **Precision-Recall AUC** (especially for heavily imbalanced data)
- `class_weight='balanced'` in sklearn models
- Stratified train/test splits

Never use raw accuracy as your primary metric for imbalanced time series.

---

## Step 5 — Visual Verification (Never Skip This)

Always plot at least one labeled window over the raw signal to confirm labels are correct:

```python
import matplotlib.pyplot as plt

event = event_windows[0]
start = pd.Timestamp(event["start"])
end   = pd.Timestamp(event["end"])

# Show a window around the event (e.g., 3 days before and after)
plot_df = df[
    (df["timestamp"] >= start - pd.Timedelta(days=3)) &
    (df["timestamp"] <= end   + pd.Timedelta(days=3))
]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(plot_df["timestamp"], plot_df["your_signal_column"])
ax.axvspan(start, end, alpha=0.3, color="red", label="event")
ax.legend()
plt.tight_layout()
plt.show()
```

**What to look for:**
- Does the signal visually change during the labeled window?
- Does the change actually start *before* the official label start? (common — technicians log *discovery*, not *onset*)
- Are there unlabeled anomalies visible elsewhere in the series?

---

## The Bigger Picture — Why Labeling Enables Everything Downstream

| Downstream task | Why it needs labels |
|---|---|
| Supervised classification / regression | Labels are the training target |
| Model evaluation (F1, AUC, RMSE) | Metrics require ground truth |
| Lead-time / early warning analysis | Measure how early the model detects the event |
| Backtesting an alert system | Distinguish true positives from false alarms |
| Concept drift detection | Know when distribution shift is due to real degradation |
| Remaining Useful Life (RUL) estimation | Time-to-failure requires knowing when failure occurred |

---

## Extension: Loading Windows from a File

For larger projects, store event windows in a CSV instead of hardcoding:

```python
# events.csv format:
# name,start,end
# failure_1,2020-04-18 00:00:00,2020-04-18 23:59:00

import pandas as pd

events_df = pd.read_csv("events.csv")
event_windows = events_df.to_dict(orient="records")
# Then use the same labeling loop as above
```

This is preferred when:
- The number of events is large (dozens or hundreds)
- Events are maintained by a separate team
- You want to version-control events separately from code

---

## Common Mistakes and How to Avoid Them

| Mistake | Consequence | Fix |
|---|---|---|
| No default initialization before loop | `NaN` in non-event rows, breaks models | Always set `df["label"] = 0` before the loop |
| Raw string comparison to datetime column | Silent zero matches (no labels applied) | Always use `pd.Timestamp()` |
| Using strict `>` / `<` at boundaries | Silently drops boundary rows | Use `>=` and `<=` |
| Row loop with `iterrows()` | 100–1000x slower than vectorized | Use `df.loc[mask]` |
| Using accuracy as primary metric | Misleadingly high due to class imbalance | Use F1, precision-recall AUC |
| Skipping visual verification | Mislabeled windows corrupt training silently | Always plot at least one window |
| Timezone-naive vs. timezone-aware mismatch | Zero labels applied | Normalize all timestamps to UTC or all to naive |
| Labels from discovery time, not onset | Model learns too short a lead time | Adjust window start backward if onset is known |

---

## Key Assumptions — Be Explicit About These

| Assumption | What it means | What to do if wrong |
|---|---|---|
| Event timestamps are accurate | Labels correctly identify the right rows | Cross-check with signal plots; adjust windows |
| Timestamp column is sorted and tz-consistent | Mask comparisons work correctly | Sort with `df.sort_values("timestamp")`; unify timezones |
| Events are discrete with clear boundaries | Simple interval mask is sufficient | Add pre-event warning zone or use soft labels |
| Data outside event windows is truly normal | Non-labeled rows are clean "normal" examples | Use domain knowledge to exclude post-failure "recovery" rows |
| Class imbalance is acceptable | Model can still generalize | Use resampling, `class_weight`, or specialized loss functions |

---

## Reusable Template

```python
import pandas as pd

# ── 1. Define events ──────────────────────────────────────────────────────────
event_windows = [
    {"name": "event_A", "start": "YYYY-MM-DD HH:MM:SS", "end": "YYYY-MM-DD HH:MM:SS"},
    # or load from CSV: pd.read_csv("events.csv").to_dict(orient="records")
]

# ── 2. Initialize defaults ────────────────────────────────────────────────────
df["label"]      = 0
df["label_name"] = "normal"

# ── 3. Apply labels ───────────────────────────────────────────────────────────
for event in event_windows:
    start = pd.Timestamp(event["start"])
    end   = pd.Timestamp(event["end"])
    mask  = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df.loc[mask, "label"]      = 1
    df.loc[mask, "label_name"] = event["name"]

# ── 4. Sanity check ───────────────────────────────────────────────────────────
print("Labeled rows:", df["label"].sum(), "/", len(df))
print(df["label"].value_counts())

# ── 5. Visual check (at least one window) ────────────────────────────────────
# (see Step 5 above)

# ── 6. Save ───────────────────────────────────────────────────────────────────
df.to_parquet("data/processed/labeled.parquet", index=False)
```
