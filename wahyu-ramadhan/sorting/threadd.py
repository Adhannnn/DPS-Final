import pandas as pd
from pathlib import Path
import threading, time

main_dir = Path(__file__).resolve().parents[2]
csv_file = main_dir / "train.csv"

def sorting_trip_duration(data):
    return data.sort_values("trip_duration")

def thread_worker(func, data, result, idx):
    result[idx] = func(data)

def time_func(func, data):
    result = [None]
    thread = threading.Thread(target=thread_worker, args=(func, data, result, 0))
    start = time.perf_counter()
    thread.start()
    thread.join()
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

    print("[Threading Sorting]")
    for label, data in splits.items():
        t = time_func(sorting_trip_duration, data)
        print(f"{label} data : {t:.4f} seconds")