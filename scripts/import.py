import os
from pymongo import MongoClient
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.column_names = pd.read_csv(self.file_path, nrows=0).columns.tolist()

    def load_data(self):
        dtypes = {
            "# Timestamp": "object",
            "MMSI": "int64",  
            "Latitude": "float64",  
            "Longitude": "float64",  
            "ROT": "float64",  
            "SOG": "float64",  
            "COG": "float64",
        }
        with ProgressBar():
            df = dd.read_csv(
                self.file_path,
                blocksize="75MB",
                dtype=dtypes,
                assume_missing=True,
                usecols=["# Timestamp", "MMSI", "Latitude", "Longitude", "ROT", "SOG", "COG"]
            )
            df_cleaned = df.dropna(subset=['Latitude', 'Longitude', '# Timestamp'])
            df_cleaned = df_cleaned.drop_duplicates()
            result = df_cleaned.compute()
        print(f"Dataset loaded: {result.shape[0]} rows")
        return result

    def export_to_mongo(self, df, db_name='TASK_3', coll_name='BIG_DATA'):
        db_url = os.getenv("DB_URL", "localhost")
        db_port = int(os.getenv("DB_PORT", 27017))
        username = os.getenv("DB_USER", "admin")
        password = os.getenv("DB_PASS", "BigData")
        uri = f"mongodb://{username}:{password}@{db_url}:{db_port}/"

        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()
        except Exception as e:
            print("MongoDB connection failed:", e)
            return
        
        
        db = client[db_name]
        collection = db[coll_name]
        if collection.estimated_document_count() > 0:
            print("Data already exists in MongoDB. Skipping import.")
            return

        batch_size = 10000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].to_dict(orient='records')
            if batch:
                collection.insert_many(batch)
                print(f"Inserted batch {i // batch_size + 1}")

        print("Data import complete.")

if __name__ == '__main__':
    csv_path = os.getenv("CSV_PATH", "aisdk-2024-05-01.csv")
    loader = DataLoader(csv_path)
    cleaned_df = loader.load_data()
    loader.export_to_mongo(cleaned_df)
