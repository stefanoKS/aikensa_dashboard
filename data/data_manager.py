# data/data_manager.py
import os
import pandas as pd
from datetime import datetime
from data.db_handler import DatabaseHandler

class DataManager:
    def __init__(self,
                 db_handler: DatabaseHandler,
                 cache_path: str = "cache/processed.parquet"):
        self.db = db_handler
        self.cache_path = cache_path

    def load_cache(self) -> pd.DataFrame:
        """Load the local cache, or return empty DF if not exists."""
        if os.path.exists(self.cache_path):
            return pd.read_parquet(self.cache_path)
        return pd.DataFrame()

    def get_max_timestamp(self, df: pd.DataFrame) -> datetime:
        """Get the max full_timestamp from the cache."""
        if df.empty or 'full_timestamp' not in df:
            # if you have no cache, start from epoch or a safe date
            return datetime(1970, 1, 1)
        return df['full_timestamp'].max()

    def fetch_and_update(self) -> pd.DataFrame:
        """
        - Loads existing cache.
        - Fetches only rows newer than last timestamp.
        - Preprocesses them.
        - Appends to cache, drops duplicates, saves.
        - Returns the up‑to‑date DataFrame.
        """
        # 1) load cache
        cache_df = self.load_cache()
        last_ts = self.get_max_timestamp(cache_df)

        # 2) build an incremental query
        #    assumes you have a timestampDate column in YYYYMMDD format
        date_str = last_ts.strftime("%Y%m%d")
        query = (
            "SELECT * FROM AIKENSAresults.inspection_results "
            f"WHERE timestampDate >= '{date_str}'"
        )

        # 3) fetch & preprocess just the delta
        raw_new = self.db.fetch_data(query)
        if raw_new.empty:
            # nothing new
            return cache_df

        processed_new = self.db.preprocess_data(raw_new)
        if processed_new.empty:
            return cache_df

        # 4) append, dedupe, sort
        combined = pd.concat([cache_df, processed_new], ignore_index=True)
        # dedupe by your unique key (e.g. an 'id' column or full_timestamp+partName)
        combined.drop_duplicates(
            subset=['partName', 'full_timestamp', 'kensainName'],
            keep='last', inplace=True)
        combined.sort_values(['partName', 'full_timestamp'], inplace=True)
        # 5) save back
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        combined.to_parquet(self.cache_path, index=False)
        return combined
