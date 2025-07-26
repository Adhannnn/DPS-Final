import pandas as pd
import os

df = pd.read_csv("train.csv", usecols=["trip_duration"])

output_dir = "splits" # Folder to save split dataset's file
os.makedirs(output_dir, exist_ok=True)

splits = {
    "25": df.iloc[:int(0.25 * len(df))],
    "50": df.iloc[:int(0.50 * len(df))],
    "75": df.iloc[:int(0.75 * len(df))],
    "100": df
}

for key, subset in splits.items(): # Loop for save dataset for each scale
    filename = f"{output_dir}/dataset_split_{key}percent.csv"
    subset.to_csv(filename, index=False)
    print(f"Saved: {filename} - shape: {subset.shape}")
