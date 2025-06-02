import os
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_data_from_mongo(batch_limit=50000):
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:BigData@mongo:27017/")
    client = MongoClient(mongo_uri)
    db = client["TASK_3"]
    coll = db["BIG_DATA"]

    print("[INFO] Streaming data from MongoDB in batches...")

    cursor = coll.find({}, {
        "_id": 0,
        "MMSI": 1,
        "# Timestamp": 1
    }).batch_size(1000)

    batch = []
    total = 0
    dataframes = []

    for doc in cursor:
        batch.append(doc)
        if len(batch) >= batch_limit:
            df = pd.DataFrame(batch)
            df["# Timestamp"] = pd.to_datetime(df["# Timestamp"], errors='coerce')
            df = df.dropna(subset=["# Timestamp"])
            dataframes.append(df)
            total += len(df)
            batch.clear()

    # Final batch
    if batch:
        df = pd.DataFrame(batch)
        df["# Timestamp"] = pd.to_datetime(df["# Timestamp"], errors='coerce')
        df = df.dropna(subset=["# Timestamp"])
        dataframes.append(df)
        total += len(df)

    print(f"[INFO] Streamed and cleaned {total} rows.")
    return pd.concat(dataframes, ignore_index=True)


def compute_delta_for_group(group):
    group = group.sort_values("# Timestamp")
    delta = group["# Timestamp"].diff().dropna().dt.total_seconds() * 1000
    return delta


def compute_delta_t_parallel(df, min_points=100):
    grouped = df.groupby("MMSI")
    filtered_groups = [group for _, group in grouped if len(group) >= min_points]

    print(f"[INFO] Processing {len(filtered_groups)} MMSIs in parallel using {cpu_count()} cores...")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_delta_for_group, filtered_groups),
                            total=len(filtered_groups), desc="Computing Δt"))

    combined = pd.concat(results, ignore_index=True)
    print(f"[INFO] Total Δt values computed: {len(combined):,}")
    return combined


def plot_delta_t_histogram(delta_t_series, bins=100):
    if delta_t_series.empty:
        print("[WARN] No data to plot.")
        return

    xmax = int(delta_t_series.max())
    plt.figure(figsize=(10, 6))
    plt.hist(delta_t_series, bins=bins, edgecolor='black')
    plt.yscale('log')
    plt.title(f'Histogram of Time Differences (Δt) in Milliseconds (≤ {xmax:,} ms)')
    plt.xlabel('Delta t (ms)')
    plt.ylabel('Frequency (log scale)')
    plt.xlim(0, xmax)
    plt.grid(True, which='both', axis='both')
    plt.tight_layout()
    plt.savefig("delta_t_histogram.png")
    print("[INFO] Histogram saved to delta_t_histogram.png")


if __name__ == "__main__":
    df = get_data_from_mongo()
    delta_t_values = compute_delta_t_parallel(df, min_points=100)
    plot_delta_t_histogram(delta_t_values)


