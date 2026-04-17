# data/data_manager.py
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.db_handler import DatabaseHandler


logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self,
                 db_handler: DatabaseHandler,
                 cache_path: str = "cache/processed.parquet"):
        self.db = db_handler
        self.cache_path = cache_path

    def _quarantine_cache(self, reason: str) -> None:
        cache_file = Path(self.cache_path)
        if not cache_file.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_path = cache_file.with_suffix(f"{cache_file.suffix}.corrupt_{timestamp}")
        try:
            cache_file.rename(quarantine_path)
            logger.warning("Quarantined cache file %s -> %s (%s)", cache_file, quarantine_path, reason)
        except OSError as exc:
            logger.warning("Failed to quarantine cache file %s (%s): %s", cache_file, reason, exc)

    def _write_cache_atomically(self, df: pd.DataFrame) -> None:
        cache_file = Path(self.cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        temp_file = cache_file.with_suffix(f"{cache_file.suffix}.tmp")
        try:
            df.to_parquet(temp_file, index=False)
            if not temp_file.exists() or temp_file.stat().st_size == 0:
                raise IOError(f"Temporary parquet write failed for {temp_file}")
            os.replace(temp_file, cache_file)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def load_cache(self) -> pd.DataFrame:
        """Load the local cache, or return empty DF if not exists."""
        cache_file = Path(self.cache_path)
        if cache_file.exists():
            if cache_file.stat().st_size == 0:
                self._quarantine_cache("cache file is 0 bytes")
                return pd.DataFrame()

            try:
                return pd.read_parquet(cache_file)
            except Exception as exc:
                self._quarantine_cache(str(exc))
                logger.warning("Failed to read cache %s, starting from empty cache: %s", cache_file, exc)
                return pd.DataFrame()
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
        self._write_cache_atomically(combined)
        return combined
