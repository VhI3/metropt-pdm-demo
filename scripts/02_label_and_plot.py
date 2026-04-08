# scripts/02_label_and_plot.py

import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


raw_folder = Path(__file__).parent.parent / "data" / "raw"
parquet_files = list(raw_folder.glob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError("No Parquet file found in data/raw/")

parquet_path = parquet_files[0]
print("Using file:", parquet_path)

df = pd.read_parquet(parquet_path)

failure_windows = [
    {"name": "failure_1", "start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:00"},
    {"name": "failure_2", "start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00"},
    {"name": "failure_3", "start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00"},
    {"name": "failure_4", "start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00"},
]

df["failure"] = 0
df["failure_name"] = "none"

for event in failure_windows:
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df.loc[mask, "failure"] = 1
    df.loc[mask, "failure_name"] = event["name"]

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/processed/metropt_labeled.parquet", index=False)

print("Saved labeled file: data/processed/metropt_labeled.parquet")
print("Number of failure rows:", df["failure"].sum())

columns_to_plot = ["tp2", "tp3", "oil_temperature", "motor_current"]

event = failure_windows[0]
start = pd.Timestamp(event["start"])
end = pd.Timestamp(event["end"])

window_start = start - pd.Timedelta(days=3)
window_end = end + pd.Timedelta(days=3)

small = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)]

Path("plots").mkdir(parents=True, exist_ok=True)

for col in columns_to_plot:
    if col not in df.columns:
        print(f"Skipping {col} because it does not exist.")
        continue

    plt.figure(figsize=(14, 4))
    plt.plot(small["timestamp"], small[col])
    plt.axvspan(start, end, alpha=0.3)
    plt.title(f"{col} around {event['name']}")
    plt.xlabel("timestamp")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"plots/{col}_around_{event['name']}.png", dpi=150)
    plt.close()

print("Plots saved in ./plots")