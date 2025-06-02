import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor
from data_loader import load_data

COLUMNS_TO_CHECK = ['MMSI', 'Navigational status', 'Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading']


# Function to clean and filter a chunk of data, annotated with the thread (CPU) index
def clean_and_filter_chunk(chunk, min_points, cpu_id):
    print(f"[CPU {cpu_id}] Starting processing with {len(chunk)} rows...")

    # Step 1: Drop missing values
    initial_row_count = len(chunk)
    chunk = chunk.dropna(subset=COLUMNS_TO_CHECK)
    dropped_na = initial_row_count - len(chunk)
    if dropped_na > 0:
        print(f"[CPU {cpu_id}] Dropped {dropped_na} rows due to missing values.")

    # Step 2: Remove irrational values
    lat_invalid = ~chunk['Latitude'].between(-90, 90)
    lon_invalid = ~chunk['Longitude'].between(-180, 180)
    rot_invalid = ~chunk['ROT'].between(-720, 720)
    cog_invalid = ~chunk['COG'].between(0, 360)
    sog_invalid = ~chunk['SOG'].between(0, 102.2)
    total_invalid = lat_invalid | lon_invalid | rot_invalid | cog_invalid | sog_invalid
    dropped_invalid = total_invalid.sum()
    chunk = chunk[~total_invalid]
    if dropped_invalid > 0:
        print(f"[CPU {cpu_id}] Dropped {dropped_invalid} rows due to irrational values.")

    # Step 3: Filter MMSIs with sufficient data points
    mmsi_counts = chunk['MMSI'].value_counts()
    initial_mmsi_count = len(mmsi_counts)
    valid_mmsis = mmsi_counts[mmsi_counts >= min_points].index
    chunk = chunk[chunk['MMSI'].isin(valid_mmsis)]
    dropped_mmsi = initial_mmsi_count - len(chunk['MMSI'].value_counts())
    if dropped_mmsi > 0:
        print(f"[CPU {cpu_id}] Dropped {dropped_mmsi} MMSIs with fewer than {min_points} points.")

    print(f"[CPU {cpu_id}] Finished processing. Final row count: {len(chunk)}")
    return chunk


# Function to parallelize the cleaning and filtering process
def parallel_clean_and_filter(df, min_points=100):
    num_cpus = 2
    print(f"Running on {num_cpus} threads...")

    # Step 1: Split the DataFrame by unique MMSIs
    mmsi_list = df['MMSI'].unique()
    chunk_size = math.ceil(len(mmsi_list) / num_cpus)
    mmsi_chunks = [mmsi_list[i:i + chunk_size] for i in range(0, len(mmsi_list), chunk_size)]

    # Step 2: Process each chunk in parallel
    cleaned_chunks = []
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = [
            executor.submit(clean_and_filter_chunk, df[df['MMSI'].isin(mmsi_chunk)], min_points, i + 1)
            for i, mmsi_chunk in enumerate(mmsi_chunks)
        ]
        for i, future in enumerate(futures):
            chunk = future.result()
            cleaned_chunks.append(chunk)
            print(f"[CPU {i+1}] Chunk completed and appended.")

    # Step 3: Combine all cleaned chunks
    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    print(f"\nFinal cleaned data has {len(cleaned_df)} rows.")
    return cleaned_df
