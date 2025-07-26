import pandas as pd
import time
import threading
import multiprocessing
from statistics import mean
import os
import numpy as np
import matplotlib.pyplot as plt

def sort_fn(queue, data):
    start = time.perf_counter()
    data.sort_values(by='trip_duration')
    queue.put(time.perf_counter() - start)

def filter_fn(queue, data):
    start = time.perf_counter()
    data[data['trip_duration'] > 1000]
    queue.put(time.perf_counter() - start)

def sort_fn_shuffle(queue, data):
    start = time.perf_counter()
    data.sort_values(by='trip_duration')
    queue.put(time.perf_counter() - start)

def filter_fn_shuffle(queue, data):
    start = time.perf_counter()
    data[data['trip_duration'] > 1000]
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

def multiprocessing_process_shuffle(df):
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=sort_fn_shuffle, args=(q, df))
    p2 = multiprocessing.Process(target=filter_fn_shuffle, args=(q, df))

    p1.start()
    p2.start()
    sort_time = q.get()
    filter_time = q.get()

    p1.join()
    p2.join()

    return sort_time, filter_time

def sequential_process(df):
    start = time.perf_counter()
    df.sort_values(by='trip_duration')
    sort_time = time.perf_counter() - start

    start = time.perf_counter()
    df[df['trip_duration'] > 1000]
    filter_time = time.perf_counter() - start

    return sort_time, filter_time

def sequential_process_shuffle(df):
    start = time.perf_counter()
    df.sort_values(by='trip_duration')
    sort_time = time.perf_counter() - start

    start = time.perf_counter()
    df[df['trip_duration'] > 1000]
    filter_time = time.perf_counter() - start

    return sort_time, filter_time

def threaded_process(df):
    sort_time = filter_time = 0.0

    def sort_fn():
        nonlocal sort_time
        start = time.perf_counter()
        df.sort_values(by='trip_duration')
        sort_time = time.perf_counter() - start

    def filter_fn():
        nonlocal filter_time
        start = time.perf_counter()
        df[df['trip_duration'] > 1000]
        filter_time = time.perf_counter() - start

    t1 = threading.Thread(target=sort_fn)
    t2 = threading.Thread(target=filter_fn)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return sort_time, filter_time

def threaded_process_shuffle(df):
    sort_time = filter_time = 0.0

    def sort_fn():
        nonlocal sort_time
        start = time.perf_counter()
        df.sort_values(by='trip_duration')
        sort_time = time.perf_counter() - start

    def filter_fn():
        nonlocal filter_time
        start = time.perf_counter()
        df[df['trip_duration'] > 1000]
        filter_time = time.perf_counter() - start

    t1 = threading.Thread(target=sort_fn)
    t2 = threading.Thread(target=filter_fn)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return sort_time, filter_time

def warm_up(df, df_shuffled, approaches):
    for _ in range(2):
        for approach in approaches:
            if 'shuffle' in approach.__name__:
                approach(df_shuffled.copy())
            else:
                approach(df.copy())

def plot_averages(results, splits):
    sizes = list(splits.keys())
    approaches = ['sequential_process', 'threaded_process', 'multiprocessing_process']
    approaches_shuffle = ['sequential_process_shuffle', 'threaded_process_shuffle', 'multiprocessing_process_shuffle']
    approach_names = ['Sequential', 'Threading', 'Multiprocessing']

    avg_sort_non_shuffle = {app: [mean(results[size][app]['sort']) for size in sizes] for app in approaches}
    avg_sort_shuffle = {app: [mean(results[size][app]['sort']) for size in sizes] for app in approaches_shuffle}
    avg_filter_non_shuffle = {app: [mean(results[size][app]['filter']) for size in sizes] for app in approaches}
    avg_filter_shuffle = {app: [mean(results[size][app]['filter']) for size in sizes] for app in approaches_shuffle}

    plt.figure(figsize=(8, 5))
    for i, app in enumerate(approaches):
        plt.plot(sizes, avg_sort_non_shuffle[app], marker='o', label=approach_names[i])
    plt.title('Average Sorting Time (Non-Shuffle)')
    plt.xlabel('Data Size')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_sort_non_shuffle.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, app in enumerate(approaches_shuffle):
        plt.plot(sizes, avg_sort_shuffle[app], marker='s', label=approach_names[i])
    plt.title('Average Sorting Time (Shuffle)')
    plt.xlabel('Data Size')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_sort_shuffle.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, app in enumerate(approaches):
        plt.plot(sizes, avg_filter_non_shuffle[app], marker='o', label=approach_names[i])
    plt.title('Average Filtering Time (Non-Shuffle)')
    plt.xlabel('Data Size')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_filter_non_shuffle.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, app in enumerate(approaches_shuffle):
        plt.plot(sizes, avg_filter_shuffle[app], marker='s', label=approach_names[i])
    plt.title('Average Filtering Time (Shuffle)')
    plt.xlabel('Data Size')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_filter_shuffle.png')
    plt.close()

if __name__ == "__main__":
    csv_file = "train/train.csv"
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        exit(1)

    try:
        df_full = pd.read_csv(csv_file)
        if 'trip_duration' not in df_full.columns:
            print("Error: Column 'trip_duration' not found in CSV.")
            exit(1)
        df_full = df_full[['trip_duration']]
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    splits = {
        '25%': df_full.iloc[:int(0.25 * len(df_full))].copy(),
        '50%': df_full.iloc[:int(0.50 * len(df_full))].copy(),
        '75%': df_full.iloc[:int(0.75 * len(df_full))].copy(),
        '100%': df_full.copy()
    }
    splits_shuffled = {
        label: df.sample(frac=1, random_state=42).reset_index(drop=True)
        for label, df in splits.items()
    }

    num_trials = 10
    approaches = [sequential_process, sequential_process_shuffle, 
                  threaded_process, threaded_process_shuffle, 
                  multiprocessing_process, multiprocessing_process_shuffle]
    results = {label: {approach.__name__: {'sort': [], 'filter': []}
                      for approach in approaches}
              for label in splits.keys()}

    for label, subset in splits.items():
        print(f"Running warm-up for {label}...")
        warm_up(subset, splits_shuffled[label], approaches)

    for trial in range(1, num_trials + 1):
        print(f"\n=== Trial {trial} ===")
        
        print("\nNon-Shuffle Results:")
        print("-" * 60)
        print(f"{'Size':<8} {'Approach':<26} {'Sort (s)':<12} {'Filter (s)':<12}")
        print("-" * 60)

        for label, subset in splits.items():
            s_sort, s_filter = sequential_process(subset.copy())
            results[label]['sequential_process']['sort'].append(s_sort)
            results[label]['sequential_process']['filter'].append(s_filter)
            print(f"{label:<8} {'Sequential':<26} {s_sort:<12.4f} {s_filter:<12.4f}")

            t_sort, t_filter = threaded_process(subset.copy())
            results[label]['threaded_process']['sort'].append(t_sort)
            results[label]['threaded_process']['filter'].append(t_filter)
            print(f"{label:<8} {'Threading':<26} {t_sort:<12.4f} {t_filter:<12.4f}")

            m_sort, m_filter = multiprocessing_process(subset.copy())
            results[label]['multiprocessing_process']['sort'].append(m_sort)
            results[label]['multiprocessing_process']['filter'].append(m_filter)
            print(f"{label:<8} {'Multiprocessing':<26} {m_sort:<12.4f} {m_filter:<12.4f}")

        print("\nShuffle Results:")
        print("-" * 60)
        print(f"{'Size':<8} {'Approach':<26} {'Sort (s)':<12} {'Filter (s)':<12}")
        print("-" * 60)

        for label, subset in splits_shuffled.items():
            ss_sort, ss_filter = sequential_process_shuffle(subset.copy())
            results[label]['sequential_process_shuffle']['sort'].append(ss_sort)
            results[label]['sequential_process_shuffle']['filter'].append(ss_filter)
            print(f"{label:<8} {'Sequential':<26} {ss_sort:<12.4f} {ss_filter:<12.4f}")

            ts_sort, ts_filter = threaded_process_shuffle(subset.copy())
            results[label]['threaded_process_shuffle']['sort'].append(ts_sort)
            results[label]['threaded_process_shuffle']['filter'].append(ts_filter)
            print(f"{label:<8} {'Threading':<26} {ts_sort:<12.4f} {ts_filter:<12.4f}")

            ms_sort, ms_filter = multiprocessing_process_shuffle(subset.copy())
            results[label]['multiprocessing_process_shuffle']['sort'].append(ms_sort)
            results[label]['multiprocessing_process_shuffle']['filter'].append(ms_filter)
            print(f"{label:<8} {'Multiprocessing':<26} {ms_sort:<12.4f} {ms_filter:<12.4f}")

    print(f"\n=== Average of {num_trials} Trials ===")
    
    print("\nAverage Sorting (Non-Shuffle):")
    print("-" * 36)
    print(f"{'Size':<8} {'Approach':<26} {'Avg Sort (s)':<12}")
    print("-" * 36)
    for label in splits.keys():
        for approach in [sequential_process, threaded_process, multiprocessing_process]:
            avg_sort = mean(results[label][approach.__name__]['sort'])
            approach_name = approach.__name__.replace('_process', '').capitalize() 
            print(f"{label:<8} {approach_name:<26} {avg_sort:<12.4f}")

    print("\nAverage Sorting (Shuffle):")
    print("-" * 36)
    print(f"{'Size':<8} {'Approach':<26} {'Avg Sort (s)':<12}")
    print("-" * 36)
    for label in splits.keys():
        for approach in [sequential_process_shuffle, threaded_process_shuffle, multiprocessing_process_shuffle]:
            avg_sort = mean(results[label][approach.__name__]['sort'])
            approach_name = approach.__name__.replace('_process', '').capitalize()
            print(f"{label:<8} {approach_name:<26} {avg_sort:<12.4f}")

    print("\nAverage Filtering (Non-Shuffle):")
    print("-" * 36)
    print(f"{'Size':<8} {'Approach':<26} {'Avg Filter (s)':<12}")
    print("-" * 36)
    for label in splits.keys():
        for approach in [sequential_process, threaded_process, multiprocessing_process]:
            avg_filter = mean(results[label][approach.__name__]['filter'])
            approach_name = approach.__name__.replace('_process', '').capitalize() 
            print(f"{label:<8} {approach_name:<26} {avg_filter:<12.4f}")

    print("\nAverage Filtering (Shuffle):")
    print("-" * 36)
    print(f"{'Size':<8} {'Approach':<26} {'Avg Filter (s)':<12}")
    print("-" * 36)
    for label in splits.keys():
        for approach in [sequential_process_shuffle, threaded_process_shuffle, multiprocessing_process_shuffle]:
            avg_filter = mean(results[label][approach.__name__]['filter'])
            approach_name = approach.__name__.replace('_process', '').capitalize()
            print(f"{label:<8} {approach_name:<26} {avg_filter:<12.4f}")

    print("\nGenerating plots...")
    plot_averages(results, splits)
    print("Plots saved: avg_sort_non_shuffle.png, avg_sort_shuffle.png, "
          "avg_filter_non_shuffle.png, avg_filter_shuffle.png")