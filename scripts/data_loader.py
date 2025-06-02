# data_loader.py

import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import math
import time

from pymongo import MongoClient
import os


df = None
def load_data(file_path=None, from_mongo=True):
    global df
    if df is not None:
        print("Data already loaded.")
        return df

    if from_mongo:
        print("Loading data from MongoDB...")
        mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:BigData@mongo:27017/")
        client = MongoClient(mongo_uri)
        db = client["TASK_3"]
        collection = db["BIG_DATA"]
        cursor = collection.find({}, {
            "_id": 0,  # exclude MongoDB document IDs
            "MMSI": 1,
            "# Timestamp": 1,
            "Latitude": 1,
            "Longitude": 1,
            "ROT": 1,
            "SOG": 1,
            "COG": 1
        })
        df = pd.DataFrame(list(cursor))
        print(f"Loaded {len(df)} rows from MongoDB.")
    else:
        # Fallback: read from CSV if explicitly needed
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from CSV.")

    print("First few rows of the data:")
    print(df.head())
    return df


def process_chunk(chunk, index):
    # Simulate a time-consuming operation
    time.sleep(0.5)
    print(f"Process-{index}: processed {len(chunk)} rows.")


def parallel_process(df):
    num_processes = min(cpu_count(), 2)
    chunk_size = math.ceil(len(df) / num_processes)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    print(f"Starting parallel processing with {len(chunks)} processes...")

    processes = []
    for i, chunk in enumerate(chunks):
        p = Process(target=process_chunk, args=(chunk, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes completed.")
