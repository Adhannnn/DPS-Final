import pandas as pd
import multiprocessing, time
from pathlib import Path

main_dir = Path(__file__).resolve().parents[2]
csv_file = main_dir / "train.csv"

def filter_trip_duration(data, threshold=1000):
    return data[data["trip_duration"] > threshold]

def filter_gt_1000(data):
    return filter_trip_duration(data, 1000)

def multiprocessing_work(func, data, return_dict):
    return_dict["result"] = func(data)

def time_func(func, data):
    mp_manager = multiprocessing.Manager()
    return_dict = mp_manager.dict()
    p = multiprocessing.Process(target=multiprocessing_work, args=(func, data, return_dict))
    start = time.perf_counter()
    p.start()
    p.join()
    end = time.perf_counter()
    return end - start

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)  # Tambahkan ini di macOS!

    df = pd.read_csv(csv_file)[["trip_duration"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    splits = {
        "25%" : df.iloc[:int(0.25 * len(df))],
        "50%" : df.iloc[:int(0.5 * len(df))],
        "75%" : df.iloc[:int(0.75 * len(df))],
        "100%" : df
    }

    print("[Multiprocessing Filtering]")
    for label, data in splits.items():
        t = time_func(filter_gt_1000, data)
        print(f"{label} data : {t:.4f} seconds")
