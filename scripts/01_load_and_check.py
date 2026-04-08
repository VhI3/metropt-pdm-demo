# scripts/01_download_and_check.py

from pathlib import Path
import pandas as pd

raw_folder = Path(__file__).parent.parent / "data" / "raw"
csv_files = list(raw_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in data/raw/")

csv_path = csv_files[0]
print("Using file:", csv_path)

# load csv
df = pd.read_csv(csv_path)

# clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# parse timestamp column
df["timestamp"] = pd.to_datetime(df["timestamp"])

# sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

# save parquet copy based on the path of the original csv file
df.to_parquet(csv_path.with_suffix(".parquet"), index=False)

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nStart:", df["timestamp"].min())
print("End:  ", df["timestamp"].max())

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst rows:")
print(df.head())

# check real time spacing
dt = df["timestamp"].diff().dropna().dt.total_seconds()

print("\nTime gap summary in seconds:")
print(dt.describe())

print("\nMost common time gaps:")
print(dt.value_counts().head(10))
