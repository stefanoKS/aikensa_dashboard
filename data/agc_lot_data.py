from __future__ import annotations

import logging
from datetime import timedelta

import mysql.connector
import numpy as np
import pandas as pd
import yaml


class AgcLotDataProcessor:
    def __init__(
        self,
        mysql_config_path: str = "mysql/id.yaml",
        schema_name: str = "aikensa_agc",
        table_name: str = "inspection_results",
        placeholder_lots: set[str] | None = None,
        placeholder_parts: set[str] | None = None,
    ):
        self.mysql_config_path = mysql_config_path
        self.schema_name = schema_name
        self.table_name = table_name
        self.placeholder_lots = {str(value) for value in (placeholder_lots or {"0000000000", ""})}
        self.placeholder_parts = {str(value) for value in (placeholder_parts or {"0", ""})}
        self.part_name_map = {
            "5": "J59JRH",
            "6": "J59JLH",
            "7": "J30RH",
            "8": "J30LH",
        }
        self.virtual_part_groups = {
            "J59J_ALL": {
                "label": "J59J 全て",
                "members": ["5", "6"],
            },
            "J30_ALL": {
                "label": "J30 全て",
                "members": ["7", "8"],
            },
        }
        self.break_gap_threshold_min = 30.0
        self.break_gap_assumed_min = 5.0
        self.intra_lot_idle_threshold_min = 10.0
        self.intra_lot_break_assumed_min = 3.0
        self.theoretical_entry_gap_cap_min = 2.0
        self.mysql_credentials = self._load_mysql_credentials()

    def _load_mysql_credentials(self) -> dict[str, object]:
        try:
            with open(self.mysql_config_path, "r", encoding="utf-8") as file:
                credentials = yaml.safe_load(file) or {}
        except OSError as exc:
            logging.error("Unable to load MySQL credentials from %s: %s", self.mysql_config_path, exc)
            return {}

        return {
            "user": credentials.get("id"),
            "password": credentials.get("pass"),
            "host": credentials.get("host"),
            "port": int(credentials.get("port", 3306)),
        }

    def _connect(self):
        if not self.mysql_credentials:
            raise mysql.connector.Error("MySQL credentials are not configured.")

        return mysql.connector.connect(
            database=self.schema_name,
            **self.mysql_credentials,
        )

    def exists(self) -> bool:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (self.schema_name, self.table_name),
                )
                exists = bool(cursor.fetchone()[0])
                cursor.close()
                return exists
        except mysql.connector.Error as exc:
            logging.error("Unable to verify AGC table availability: %s", exc)
            return False

    def get_part_label(self, part_value: object) -> str:
        part_key = str(part_value)
        if part_key in self.virtual_part_groups:
            return self.virtual_part_groups[part_key]["label"]
        return self.part_name_map.get(part_key, part_key)

    def resolve_part_filter_values(self, part_filter: object) -> list[str] | None:
        if part_filter is None:
            return None
        part_filter_str = str(part_filter).strip()
        if not part_filter_str:
            return None
        if part_filter_str in self.virtual_part_groups:
            return list(self.virtual_part_groups[part_filter_str]["members"])
        if part_filter_str in self.part_name_map:
            return [part_filter_str]

        for group_key, group_meta in self.virtual_part_groups.items():
            if part_filter_str == group_meta["label"]:
                return list(group_meta["members"])

        for raw_value, label in self.part_name_map.items():
            if part_filter_str == label:
                return [raw_value]
        return [part_filter_str]

    def get_selectable_parts(self, available_parts: list[str]) -> list[str]:
        normalized_parts = [str(value) for value in available_parts if str(value) not in self.placeholder_parts]
        available_set = set(normalized_parts)
        if not available_set:
            return []

        selectable_parts: list[str] = []
        inserted_groups: set[str] = set()

        for part_code in sorted(available_set, key=lambda value: int(str(value))):
            for group_key, group_meta in self.virtual_part_groups.items():
                if group_key in inserted_groups:
                    continue
                members = group_meta["members"]
                if part_code == members[0] and any(member in available_set for member in members):
                    selectable_parts.append(group_key)
                    inserted_groups.add(group_key)
            selectable_parts.append(part_code)

        return selectable_parts

    def _read_sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        if not self.exists():
            return pd.DataFrame()

        try:
            with self._connect() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(query, params or [])
                rows = cursor.fetchall()
                cursor.close()
        except mysql.connector.Error as exc:
            logging.error("Unable to load AGC lot data from %s.%s: %s", self.schema_name, self.table_name, exc)
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def get_date_context(self) -> dict[str, object]:
        bounds_df = self._read_sql(
            f"""
            SELECT MIN(timestamp) AS min_timestamp, MAX(timestamp) AS max_timestamp
            FROM {self.table_name}
            """
        )
        if bounds_df.empty:
            return {}

        min_timestamp = pd.to_datetime(bounds_df.at[0, "min_timestamp"], errors="coerce")
        max_timestamp = pd.to_datetime(bounds_df.at[0, "max_timestamp"], errors="coerce")
        if pd.isna(min_timestamp) or pd.isna(max_timestamp):
            return {}

        min_date = min_timestamp.date()
        max_date = max_timestamp.date()
        default_end_date = max_date
        default_start_date = max(min_date, default_end_date - timedelta(days=6))

        return {
            "min_date": min_date - timedelta(days=5),
            "max_date": max_date + timedelta(days=5),
            "default_start_date": default_start_date,
            "default_end_date": default_end_date,
            "data_min_date": min_date,
            "data_max_date": max_date,
            "data_min_timestamp": min_timestamp,
            "data_max_timestamp": max_timestamp,
        }

    def get_available_parts(self, start_date: object = None, end_date: object = None) -> list[str]:
        query = f"SELECT DISTINCT partName FROM {self.table_name}"
        params: list[object] = []
        filters: list[str] = []

        if start_date:
            filters.append("timestamp >= %s")
            params.append(f"{pd.to_datetime(start_date).date()} 00:00:00")
        if end_date:
            filters.append("timestamp <= %s")
            params.append(f"{pd.to_datetime(end_date).date()} 23:59:59")

        if filters:
            query = f"{query} WHERE {' AND '.join(filters)}"
        query = f"{query} ORDER BY partName"

        parts_df = self._read_sql(query, params)
        if parts_df.empty or "partName" not in parts_df.columns:
            return []

        return [
            str(value)
            for value in parts_df["partName"].dropna().tolist()
            if str(value) not in self.placeholder_parts
        ]

    def load_records(
        self,
        part_filter: str | None = None,
        start_date: object = None,
        end_date: object = None,
    ) -> pd.DataFrame:
        query = (
            "SELECT id, partName, lotNumber, serialNumber, ok_add, ng_add, timestamp, kensainName "
            f"FROM {self.table_name}"
        )
        params: list[object] = []
        filters: list[str] = []

        normalized_part_filters = self.resolve_part_filter_values(part_filter)

        if normalized_part_filters:
            filters.append(f"partName IN ({', '.join(['%s'] * len(normalized_part_filters))})")
            params.extend(int(part_value) for part_value in normalized_part_filters)
        if start_date:
            filters.append("timestamp >= %s")
            params.append(f"{pd.to_datetime(start_date).date()} 00:00:00")
        if end_date:
            filters.append("timestamp <= %s")
            params.append(f"{pd.to_datetime(end_date).date()} 23:59:59")

        if filters:
            query = f"{query} WHERE {' AND '.join(filters)}"
        query = f"{query} ORDER BY timestamp ASC"

        records_df = self._read_sql(query, params)
        if records_df.empty:
            return records_df

        records_df["timestamp_dt"] = pd.to_datetime(records_df["timestamp"], errors="coerce")
        records_df = records_df.dropna(subset=["timestamp_dt"]).copy()
        records_df["partName"] = records_df["partName"].astype(str)
        records_df["partLabel"] = records_df["partName"].map(self.part_name_map).fillna(records_df["partName"])
        records_df["lotNumber"] = records_df["lotNumber"].fillna("").astype(str)
        return records_df

    def _build_lot_cycle_df(self, records_df: pd.DataFrame) -> pd.DataFrame:
        if records_df.empty:
            return pd.DataFrame(columns=["inspection_date", "lotNumber", "lot_min_cycle_sec"])

        benchmark_source_df = records_df.copy()
        if "inspection_date" not in benchmark_source_df.columns:
            benchmark_source_df["inspection_date"] = benchmark_source_df["timestamp_dt"].dt.date

        lot_cycle_rows: list[dict[str, object]] = []
        for (inspection_date, lot_number), lot_df in benchmark_source_df.groupby(["inspection_date", "lotNumber"], sort=True):
            lot_df = lot_df.sort_values("timestamp_dt").reset_index(drop=True)
            if lot_df.empty:
                continue

            ideal_mask = (lot_df["ok_add"].fillna(0) == 5) & (lot_df["ng_add"].fillna(0) == 0)
            ideal_flags = ideal_mask.tolist()
            timestamps = lot_df["timestamp_dt"].tolist()
            cycle_candidates: list[float] = []

            for start_index in range(len(lot_df) - 2):
                if not (ideal_flags[start_index] and ideal_flags[start_index + 1] and ideal_flags[start_index + 2]):
                    continue

                first_delta = (timestamps[start_index + 1] - timestamps[start_index]).total_seconds()
                second_delta = (timestamps[start_index + 2] - timestamps[start_index + 1]).total_seconds()
                if first_delta > 0:
                    cycle_candidates.append(first_delta)
                if second_delta > 0:
                    cycle_candidates.append(second_delta)

            if cycle_candidates:
                lot_cycle_rows.append(
                    {
                        "inspection_date": inspection_date,
                        "lotNumber": lot_number,
                        "lot_min_cycle_sec": min(cycle_candidates),
                    }
                )

        return pd.DataFrame(lot_cycle_rows)

    def _build_daily_adjusted_elapsed_df(self, active_records_df: pd.DataFrame) -> pd.DataFrame:
        if active_records_df.empty:
            return pd.DataFrame(columns=["inspection_date", "adjusted_elapsed_sec"])

        elapsed_df = active_records_df.copy()
        elapsed_df["inspection_date"] = elapsed_df["timestamp_dt"].dt.date
        elapsed_df = elapsed_df.sort_values(["inspection_date", "timestamp_dt"]).reset_index(drop=True)
        elapsed_df["next_timestamp_dt"] = elapsed_df.groupby("inspection_date")["timestamp_dt"].shift(-1)
        elapsed_df["next_lotNumber"] = elapsed_df.groupby("inspection_date")["lotNumber"].shift(-1)
        elapsed_df["gap_sec"] = (elapsed_df["next_timestamp_dt"] - elapsed_df["timestamp_dt"]).dt.total_seconds()
        elapsed_df["adjusted_gap_sec"] = elapsed_df["gap_sec"].where(elapsed_df["gap_sec"] > 0, 0).fillna(0)

        break_mask = (
            elapsed_df["next_timestamp_dt"].notna()
            & (elapsed_df["lotNumber"] != elapsed_df["next_lotNumber"])
            & ((elapsed_df["gap_sec"] / 60.0) > self.break_gap_threshold_min)
        )
        elapsed_df.loc[break_mask, "adjusted_gap_sec"] = self.break_gap_assumed_min * 60.0

        intra_break_mask = (
            elapsed_df["next_timestamp_dt"].notna()
            & (elapsed_df["lotNumber"] == elapsed_df["next_lotNumber"])
            & ((elapsed_df["gap_sec"] / 60.0) > self.break_gap_threshold_min)
        )
        elapsed_df.loc[intra_break_mask, "adjusted_gap_sec"] = self.intra_lot_break_assumed_min * 60.0

        return (
            elapsed_df.groupby("inspection_date", as_index=False)
            .agg(adjusted_elapsed_sec=("adjusted_gap_sec", "sum"))
            .sort_values("inspection_date")
        )

    def build_daily_finished_parts(self, records_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if records_df.empty:
            empty = pd.DataFrame(columns=["inspection_date", "partName", "finished_parts"])
            return empty, pd.DataFrame()

        daily_df = records_df.copy()
        daily_df = daily_df[~daily_df["partName"].isin(self.placeholder_parts)].copy()
        if daily_df.empty:
            empty = pd.DataFrame(columns=["inspection_date", "partName", "finished_parts"])
            return empty, pd.DataFrame()

        daily_df["inspection_date"] = daily_df["timestamp_dt"].dt.date
        daily_df["finished_parts"] = daily_df["ok_add"].fillna(0) + daily_df["ng_add"].fillna(0)

        grouped_df = (
            daily_df.groupby(["inspection_date", "partName"], as_index=False)
            .agg(
                finished_parts=("finished_parts", "sum"),
                ok_total=("ok_add", "sum"),
                ng_total=("ng_add", "sum"),
                lots=("lotNumber", "nunique"),
            )
            .sort_values(["inspection_date", "partName"])
        )
        grouped_df["partLabel"] = grouped_df["partName"].map(self.part_name_map).fillna(grouped_df["partName"])

        pivot_df = (
            grouped_df.pivot(index="inspection_date", columns="partName", values="finished_parts")
            .fillna(0)
            .sort_index()
        )
        if not pivot_df.empty:
            pivot_df = pivot_df.reindex(sorted(pivot_df.columns, key=lambda value: int(str(value))), axis=1)
            pivot_df = pivot_df.rename(columns=self.part_name_map)

        return grouped_df, pivot_df.reset_index()

    def build_daily_hourly_summary(self, records_df: pd.DataFrame) -> pd.DataFrame:
        if records_df.empty:
            return pd.DataFrame(
                columns=[
                    "inspection_date",
                    "finished_parts",
                    "ok_total",
                    "ng_total",
                    "ng_rate_pct",
                    "elapsed_hours",
                    "parts_per_hour",
                    "benchmark_cycle_sec",
                    "benchmark_lot_count",
                    "theoretical_parts_per_hour",
                    "first_time",
                    "last_time",
                ]
            )

        daily_df = records_df.copy()
        daily_df = daily_df[~daily_df["partName"].isin(self.placeholder_parts)].copy()
        if daily_df.empty:
            return pd.DataFrame(
                columns=[
                    "inspection_date",
                    "finished_parts",
                    "ok_total",
                    "ng_total",
                    "ng_rate_pct",
                    "elapsed_hours",
                    "parts_per_hour",
                    "benchmark_cycle_sec",
                    "benchmark_lot_count",
                    "theoretical_parts_per_hour",
                    "first_time",
                    "last_time",
                ]
            )

        daily_df["inspection_date"] = daily_df["timestamp_dt"].dt.date
        daily_df["finished_parts"] = daily_df["ok_add"].fillna(0) + daily_df["ng_add"].fillna(0)
        active_records_df = daily_df[
            (daily_df["ok_add"].fillna(0) > 0) | (daily_df["ng_add"].fillna(0) > 0)
        ].copy()

        summary_df = (
            daily_df.groupby("inspection_date", as_index=False)
            .agg(
                finished_parts=("finished_parts", "sum"),
                ok_total=("ok_add", "sum"),
                ng_total=("ng_add", "sum"),
                first_time=("timestamp_dt", "min"),
                last_time=("timestamp_dt", "max"),
            )
            .sort_values("inspection_date")
        )

        adjusted_elapsed_df = self._build_daily_adjusted_elapsed_df(active_records_df)
        summary_df = summary_df.merge(adjusted_elapsed_df, on="inspection_date", how="left")
        summary_df["elapsed_hours"] = summary_df["adjusted_elapsed_sec"].fillna(0) / 3600.0
        summary_df["elapsed_hours"] = summary_df["elapsed_hours"].where(summary_df["elapsed_hours"] > 0, 1 / 60)
        summary_df["parts_per_hour"] = summary_df["finished_parts"] / summary_df["elapsed_hours"]
        summary_df["ng_rate_pct"] = 100 * summary_df["ng_total"] / summary_df["finished_parts"].where(summary_df["finished_parts"] > 0)

        lot_cycle_df = self._build_lot_cycle_df(daily_df)
        if lot_cycle_df.empty:
            summary_df["benchmark_cycle_sec"] = pd.NA
            summary_df["benchmark_lot_count"] = 0
            summary_df["theoretical_parts_per_hour"] = pd.NA
            return summary_df

        benchmark_df = (
            lot_cycle_df.groupby("inspection_date", as_index=False)
            .agg(
                benchmark_cycle_sec=("lot_min_cycle_sec", "mean"),
                benchmark_lot_count=("lot_min_cycle_sec", "count"),
            )
            .sort_values("inspection_date")
        )

        summary_df = summary_df.merge(benchmark_df, on="inspection_date", how="left")
        summary_df["benchmark_lot_count"] = summary_df["benchmark_lot_count"].fillna(0).astype(int)
        summary_df["theoretical_parts_per_hour"] = (
            5 * 3600 / summary_df["benchmark_cycle_sec"].where(summary_df["benchmark_cycle_sec"] > 0)
        )
        return summary_df

    def build_lot_summary(self, records_df: pd.DataFrame) -> pd.DataFrame:
        if records_df.empty:
            return pd.DataFrame()

        records_df = records_df.copy()
        records_df["inspection_date"] = records_df["timestamp_dt"].dt.date
        active_records_df = records_df[
            (records_df["ok_add"].fillna(0) > 0) | (records_df["ng_add"].fillna(0) > 0)
        ].copy()

        # Only consider gaps between active records (ok_add>0 or ng_add>0) so that
        # leading/trailing 0-count placeholder rows do not inflate the inactive gap sum.
        active_for_gaps_df = records_df[
            (records_df["ok_add"].fillna(0) > 0) | (records_df["ng_add"].fillna(0) > 0)
        ].sort_values(["lotNumber", "timestamp_dt"]).copy()
        active_for_gaps_df["next_timestamp_dt"] = active_for_gaps_df.groupby("lotNumber")["timestamp_dt"].shift(-1)
        active_for_gaps_df["gap_sec"] = (
            active_for_gaps_df["next_timestamp_dt"] - active_for_gaps_df["timestamp_dt"]
        ).dt.total_seconds()
        _gap = active_for_gaps_df["gap_sec"].fillna(0)
        _idle_sec = self.intra_lot_idle_threshold_min * 60.0
        _break_sec = self.break_gap_threshold_min * 60.0
        _assumed_sec = self.intra_lot_break_assumed_min * 60.0
        active_for_gaps_df["inactive_gap_sec"] = np.where(
            _gap > _break_sec,
            (_gap - _assumed_sec).clip(lower=0),
            np.where(_gap > _idle_sec, _gap, 0.0),
        )
        active_for_gaps_df["intra_lot_break_sec"] = np.where(
            _gap > _break_sec,
            _gap,
            0.0,
        )
        lot_gap_metrics_df = (
            active_for_gaps_df.groupby("lotNumber", as_index=False)
            .agg(
                inactive_gap_sec=("inactive_gap_sec", "sum"),
                intra_lot_break_sec=("intra_lot_break_sec", "sum"),
            )
        )

        lot_theoretical_metrics_df = records_df.sort_values(["lotNumber", "id", "timestamp_dt"]).copy()
        lot_theoretical_metrics_df["next_timestamp_dt"] = lot_theoretical_metrics_df.groupby("lotNumber")["timestamp_dt"].shift(-1)
        lot_theoretical_metrics_df["theoretical_gap_sec"] = (
            lot_theoretical_metrics_df["next_timestamp_dt"] - lot_theoretical_metrics_df["timestamp_dt"]
        ).dt.total_seconds()
        lot_theoretical_metrics_df["theoretical_gap_sec"] = lot_theoretical_metrics_df["theoretical_gap_sec"].where(
            lot_theoretical_metrics_df["theoretical_gap_sec"] > 0,
            0,
        ).fillna(0)
        lot_theoretical_metrics_df["theoretical_gap_sec"] = lot_theoretical_metrics_df["theoretical_gap_sec"].clip(
            upper=self.theoretical_entry_gap_cap_min * 60.0
        )
        lot_theoretical_metrics_df = (
            lot_theoretical_metrics_df.groupby("lotNumber", as_index=False)
            .agg(theoretical_time_sec=("theoretical_gap_sec", "sum"))
        )

        lot_cycle_df = self._build_lot_cycle_df(records_df)
        daily_benchmark_df = (
            lot_cycle_df.groupby("inspection_date", as_index=False)
            .agg(benchmark_cycle_sec=("lot_min_cycle_sec", "mean"))
            if not lot_cycle_df.empty
            else pd.DataFrame(columns=["inspection_date", "benchmark_cycle_sec"])
        )

        lot_part_labels = (
            records_df.groupby("lotNumber")["partLabel"]
            .agg(lambda values: "/".join(sorted(pd.Series(values).dropna().astype(str).unique().tolist())))
            .reset_index(name="partLabel")
        )

        active_times = (
            active_records_df.groupby("lotNumber", as_index=False)
            .agg(
                active_first_time=("timestamp_dt", "min"),
                active_last_time=("timestamp_dt", "max"),
            )
            if not active_records_df.empty
            else pd.DataFrame(columns=["lotNumber", "active_first_time", "active_last_time"])
        )

        lot_summary = (
            records_df.groupby("lotNumber", as_index=False)
            .agg(
                ok_total=("ok_add", "sum"),
                ng_total=("ng_add", "sum"),
                first_time=("timestamp_dt", "min"),
                last_time=("timestamp_dt", "max"),
                rows=("id", "count"),
            )
        )
        lot_summary["inspection_date"] = lot_summary["first_time"].dt.date
        lot_summary = lot_summary.merge(lot_part_labels, on="lotNumber", how="left")
        lot_summary = lot_summary.merge(active_times, on="lotNumber", how="left")
        lot_summary = lot_summary.merge(lot_gap_metrics_df, on="lotNumber", how="left")
        lot_summary = lot_summary.merge(lot_theoretical_metrics_df, on="lotNumber", how="left")
        lot_summary = lot_summary.merge(daily_benchmark_df, on="inspection_date", how="left")
        if not lot_cycle_df.empty:
            lot_summary = lot_summary.merge(
                lot_cycle_df[["inspection_date", "lotNumber", "lot_min_cycle_sec"]],
                on=["inspection_date", "lotNumber"],
                how="left",
            )
        else:
            lot_summary["lot_min_cycle_sec"] = pd.NA
        lot_summary["lotDisplay"] = lot_summary.apply(
            lambda row: f"{row['lotNumber']} ({row['partLabel']})" if pd.notna(row["partLabel"]) and str(row["partLabel"]).strip() else str(row["lotNumber"]),
            axis=1,
        )
        lot_summary["first_time"] = lot_summary["active_first_time"].combine_first(lot_summary["first_time"])
        lot_summary["last_time"] = lot_summary["active_last_time"].combine_first(lot_summary["last_time"])

        lot_summary["total_parts"] = lot_summary["ok_total"] + lot_summary["ng_total"]
        lot_summary["raw_duration_sec"] = (lot_summary["last_time"] - lot_summary["first_time"]).dt.total_seconds()
        lot_summary["inactive_gap_sec"] = lot_summary["inactive_gap_sec"].fillna(0)
        lot_summary["duration_sec"] = (lot_summary["raw_duration_sec"] - lot_summary["inactive_gap_sec"]).clip(lower=0)
        lot_summary["duration_min"] = lot_summary["duration_sec"] / 60.0
        lot_summary["duration_h"] = lot_summary["duration_sec"] / 3600.0
        lot_summary["dekidaka_duration_sec"] = lot_summary["raw_duration_sec"].clip(lower=0)
        lot_summary["dekidaka_duration_h"] = lot_summary["dekidaka_duration_sec"] / 3600.0
        non_zero_parts = lot_summary["total_parts"].where(lot_summary["total_parts"] != 0)
        lot_summary["sec_per_part"] = lot_summary["dekidaka_duration_sec"] / non_zero_parts
        lot_summary["ng_rate_pct"] = 100 * lot_summary["ng_total"] / non_zero_parts
        lot_summary["theoretical_time_sec"] = lot_summary["theoretical_time_sec"].where(
            lot_summary["theoretical_time_sec"] > 0,
            pd.NA,
        )
        lot_summary["theoretical_time_min"] = lot_summary["theoretical_time_sec"] / 60.0
        lot_summary["real_vs_theoretical_pct"] = 100 * lot_summary["theoretical_time_sec"] / lot_summary["duration_sec"].where(
            lot_summary["duration_sec"] > 0
        )
        lot_summary["real_vs_theoretical_pct"] = lot_summary["real_vs_theoretical_pct"].clip(upper=100)

        lot_summary = lot_summary.sort_values("first_time").reset_index(drop=True)
        lot_summary["next_lot_start"] = lot_summary["active_first_time"].shift(-1).bfill()
        lot_summary["time_to_next_lot_min"] = (
            (lot_summary["next_lot_start"] - lot_summary["active_last_time"]).dt.total_seconds() / 60.0
        )
        lot_summary["is_break_to_next_lot"] = lot_summary["time_to_next_lot_min"] > self.break_gap_threshold_min

        return lot_summary[
            ~lot_summary["lotNumber"].astype(str).isin(self.placeholder_lots)
        ].copy()

    def build_display_summary(self, lot_summary: pd.DataFrame) -> pd.DataFrame:
        if lot_summary.empty:
            return pd.DataFrame(
                columns=[
                    "partLabel",
                    "lotNumber",
                    "ok_total",
                    "ng_total",
                    "total_parts",
                    "parts_per_hour",
                    "sec_per_part",
                    "first_time",
                    "last_time",
                    "duration_min",
                    "time_to_next_lot_min",
                ]
            )

        summary_df = lot_summary.copy()
        summary_df["first_time"] = summary_df["first_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        summary_df["last_time"] = summary_df["last_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        summary_df["parts_per_hour"] = summary_df["total_parts"] / summary_df["dekidaka_duration_h"].where(summary_df["dekidaka_duration_h"] > 0)
        summary_df["parts_per_hour"] = summary_df["parts_per_hour"].round(1)
        summary_df["ng_rate_pct"] = summary_df["ng_rate_pct"].round(2)
        summary_df["sec_per_part"] = summary_df["sec_per_part"].round(2)
        summary_df["intra_lot_break_sec"] = summary_df["intra_lot_break_sec"].fillna(0)
        summary_df["duration_min"] = summary_df.apply(
            lambda row: (
                f"{row['duration_min']:.1f}(BREAK {row['intra_lot_break_sec'] / 60.0:.1f})"
                if row["intra_lot_break_sec"] > 0
                else f"{row['duration_min']:.1f}"
            ),
            axis=1,
        )
        summary_df["time_to_next_lot_min"] = summary_df["time_to_next_lot_min"].apply(
            lambda value: pd.NA if pd.isna(value) else f"BREAK({value:.1f})" if value > self.break_gap_threshold_min else f"{value:.1f}"
        )

        return summary_df[
            [
                "partLabel",
                "lotNumber",
                "ok_total",
                "ng_total",
                "total_parts",
                "parts_per_hour",
                "sec_per_part",
                "first_time",
                "last_time",
                "duration_min",
                "time_to_next_lot_min",
            ]
        ]

    def get_lot_summary(
        self,
        part_filter: str | None = None,
        start_date: object = None,
        end_date: object = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        records_df = self.load_records(part_filter=part_filter, start_date=start_date, end_date=end_date)
        lot_summary = self.build_lot_summary(records_df)
        display_summary = self.build_display_summary(lot_summary)
        daily_finished_df, daily_finished_pivot = self.build_daily_finished_parts(records_df)
        daily_hourly_df = self.build_daily_hourly_summary(records_df)
        return lot_summary, display_summary, daily_finished_df, daily_finished_pivot, daily_hourly_df