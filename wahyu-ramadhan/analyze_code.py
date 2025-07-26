import pandas as pd
import time
import threading
import multiprocessing
from statistics import mean

def sort_fn(queue, data):
    start = time.perf_counter()
    _ = data.sort_values(by='trip_duration')
    queue.put(time.perf_counter() - start)

def filter_fn(queue, data):
    start = time.perf_counter()
    _ = data[data['trip_duration'] > 1000]
    queue.put(time.perf_counter() - start)

def multiprocessing_process(df):
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=sort_fn, args=(q, df))
    p2 = multiprocessing.Process(target=filter_fn, args=(q, df))

    p1.start()
    p2.start()
    sort_time = q.get()
    filter_time = q.get()

    p1.join()
    p2.join()

    return sort_time, filter_time

def sequential_process(df):
    start = time.perf_counter()
    _ = df.sort_values(by='trip_duration')
    sort_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = df[df['trip_duration'] > 1000]
    filter_time = time.perf_counter() - start

    return sort_time, filter_time

def threaded_process(df):
    sort_time = filter_time = 0.0

    def sort_inner():
        nonlocal sort_time
        start = time.perf_counter()
        _ = df.sort_values(by='trip_duration')
        sort_time = time.perf_counter() - start

    def filter_inner():
        nonlocal filter_time
        start = time.perf_counter()
        _ = df[df['trip_duration'] > 1000]
        filter_time = time.perf_counter() - start

    t1 = threading.Thread(target=sort_inner)
    t2 = threading.Thread(target=filter_inner)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return sort_time, filter_time

if __name__ == "__main__":
    df_full = pd.read_csv("train.csv")[['trip_duration']]
    df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)  # optional shuffle

    splits = {
        '25%': df_full.iloc[:int(0.25 * len(df_full))].copy(),
        '50%': df_full.iloc[:int(0.50 * len(df_full))].copy(),
        '75%': df_full.iloc[:int(0.75 * len(df_full))].copy(),
        '100%': df_full.copy()
    }

    warmup_trials = 2
    num_trials = 10
    total_trials = warmup_trials + num_trials

    results = {label: {approach: {'sort': [], 'filter': []} 
                      for approach in ['sequential', 'threading', 'multiprocessing']} 
              for label in splits.keys()}

    for trial in range(1, total_trials + 1):
        print(f"\n=== Trial {trial} ({'Warm-Up' if trial <= warmup_trials else 'Measured'}) ===")
        print(f"{'Size':<6} {'Approach':<15} {'Sort (s)':<10} {'Filter (s)':<10}")
        print("-" * 50)

        for label, subset in splits.items():
            s_sort, s_filter = sequential_process(subset.copy())
            t_sort, t_filter = threaded_process(subset.copy())
            m_sort, m_filter = multiprocessing_process(subset.copy())

            if trial > warmup_trials:
                results[label]['sequential']['sort'].append(s_sort)
                results[label]['sequential']['filter'].append(s_filter)

                results[label]['threading']['sort'].append(t_sort)
                results[label]['threading']['filter'].append(t_filter)

                results[label]['multiprocessing']['sort'].append(m_sort)
                results[label]['multiprocessing']['filter'].append(m_filter)

            print(f"{label:<6} {'Sequential':<15} {s_sort:<10.4f} {s_filter:<10.4f}")
            print(f"{label:<6} {'Threading':<15} {t_sort:<10.4f} {t_filter:<10.4f}")
            print(f"{label:<6} {'Multiprocessing':<15} {m_sort:<10.4f} {m_filter:<10.4f}")

    print(f"\n=== Average of {num_trials} Measured Trials (after {warmup_trials} warm-ups) ===")
    print(f"{'Size':<6} {'Approach':<15} {'Avg Sort (s)':<12} {'Avg Filter (s)':<12}")
    print("-" * 50)

    for label in splits.keys():
        for approach in ['sequential', 'threading', 'multiprocessing']:
            avg_sort = mean(results[label][approach]['sort'])
            avg_filter = mean(results[label][approach]['filter'])
            print(f"{label:<6} {approach.capitalize():<15} {avg_sort:<12.4f} {avg_filter:<12.4f}")
