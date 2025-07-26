import pandas as pd
from pathlib import Path
import time

main_dir = Path(__file__).resolve().parents[2]
csv_file = main_dir / "train.csv"

def filter_trip_duration(data, threshold=1000):
    return data[data["trip_duration"] > threshold]

def time_func(func, data):
    start = time.perf_counter()
    func(data)
    end = time.perf_counter()
    return end - start

if __name__ == "__main__":
    df = pd.read_csv(csv_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    splits = {
        "25%" : df.iloc[:int(0.25 * len(df))],
        "50%" : df.iloc[:int(0.5 * len(df))],
        "75%" : df.iloc[:int(0.75 * len(df))],
        "100%" : df
    }

    print("[Sequential Filtering]")
    for label, data in splits.items():
        t = time_func(lambda d: filter_trip_duration(d, 1000), data)
        print(f"{label} data : {t:.4f} seconds")