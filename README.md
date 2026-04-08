# metropt-pdm-demo

A predictive maintenance demo using the **MetroPT-3 Air Compressor Dataset** from the UCI Machine Learning Repository.  
The project covers the full pipeline: data loading, failure labeling, feature engineering, and model-ready dataset creation.

---

## Dataset

**MetroPT-3 Air Compressor Dataset**  
Source: [UCI ML Repository – MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)

- ~1.5 million rows, 1-second sampling interval
- Recorded February – August 2020 on a metro train air compressor
- 15 sensor signals (pressure, temperature, current, digital relays)
- 4 labeled air leak failure events from official company maintenance reports

The dataset is **not included** in this repository (too large for GitHub).  
Run cell 2 of the notebook to download it automatically via `wget`.

---

## Project Structure

```
notebooks/
    01_explore.ipynb   ← main notebook: EDA, labeling, feature engineering
scripts/
    01_load_and_check.py
    02_label_and_plot.py
    03_make_features.py
    04_backtest_baseline.py
    05_train_model.py
    06_detect_anomalies.py
    07_check_drift.py
data/
    raw/               ← downloaded dataset goes here (gitignored)
    processed/         ← generated feature files go here (gitignored)
requirements.txt
Note.md                ← educational cheat sheet on time series labeling
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/VhI3/metropt-pdm-demo.git
cd metropt-pdm-demo

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter lab notebooks/01_explore.ipynb
```

Then run cell 2 to download and extract the dataset automatically.

---

## What the Notebook Covers

| Section | Description |
|---|---|
| **1. Load & Check** | Load CSV, normalize columns, inspect dtypes and time gaps |
| **2. Label Failures & Plot** | Mark 4 known failure windows, visualize with red shading |
| **3. Feature Engineering** | Lag, diff, rolling mean/std features; save `metropt_features.parquet` |

---

## Key Findings

- `tp2` has a **bimodal distribution**: ~84% near zero (idle/unloaded state) and ~16% at 8–10 bar (active compression)
- Failure events are rare (~0.2% of rows) — important for choosing evaluation metrics (precision/recall over accuracy)
- Rolling window features use `shift(1)` to prevent data leakage from future values

---

## Requirements

See `requirements.txt`. Main dependencies: `pandas`, `matplotlib`, `scikit-learn`, `pyarrow`.

