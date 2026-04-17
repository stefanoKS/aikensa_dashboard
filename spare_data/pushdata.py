from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import mysql.connector
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEMP_ROOT = ROOT_DIR / "temp"
DEFAULT_CONFIG_PATH = ROOT_DIR / "mysql" / "id.yaml"
DEFAULT_SCHEMA = "aikensa_agc"
DEFAULT_TABLE = "inspection_results"
SOURCE_QUERY = """
    SELECT partName, lotNumber, serialNumber, ok_add, ng_add, timestamp, kensainName
    FROM inspection_results
    ORDER BY timestamp ASC, id ASC
"""
INSERT_QUERY = """
    INSERT IGNORE INTO inspection_results (
        partName, lotNumber, serialNumber, ok_add, ng_add, timestamp, kensainName
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""
CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS inspection_results (
      id BIGINT NOT NULL AUTO_INCREMENT,
      partName INT NOT NULL,
      lotNumber VARCHAR(255) NOT NULL,
      serialNumber VARCHAR(255) NOT NULL,
      ok_add INT NOT NULL DEFAULT 0,
      ng_add INT NOT NULL DEFAULT 0,
      timestamp DATETIME NOT NULL,
      kensainName VARCHAR(255) DEFAULT NULL,
      PRIMARY KEY (id),
      UNIQUE KEY uq_part_lot_serial (partName, lotNumber, serialNumber),
      KEY idx_part_lot_time (partName, lotNumber, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import AGC lot SQLite data from temp into MySQL.")
    parser.add_argument("--temp-root", default=str(DEFAULT_TEMP_ROOT), help="Root folder containing AGC temp databases.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="MySQL credential YAML file.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Destination MySQL schema.")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Destination MySQL table.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per batch insert.")
    return parser.parse_args()


def load_mysql_credentials(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file:
        credentials = yaml.safe_load(file) or {}
    return {
        "host": credentials["host"],
        "user": credentials["id"],
        "password": credentials["pass"],
        "port": int(credentials.get("port", 3306)),
    }


def discover_sqlite_databases(temp_root: Path) -> list[Path]:
    db_paths = sorted(temp_root.glob("**/agc_database_results.db"))
    if not db_paths:
        raise FileNotFoundError(f"No AGC SQLite databases found under {temp_root}")
    return db_paths


def read_sqlite_rows(db_path: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(SOURCE_QUERY)
        return cursor.fetchall()


def ensure_destination(mysql_conn, schema: str) -> None:
    cursor = mysql_conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {schema}")
    cursor.execute(f"USE {schema}")
    cursor.execute(CREATE_TABLE_SQL)
    mysql_conn.commit()
    cursor.close()


def import_rows(mysql_conn, schema: str, db_path: Path, rows: list[tuple[object, ...]], batch_size: int) -> tuple[int, int]:
    if not rows:
        return 0, 0

    cursor = mysql_conn.cursor()
    cursor.execute(f"USE {schema}")
    inserted = 0
    attempted = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        cursor.executemany(INSERT_QUERY, batch)
        inserted += cursor.rowcount
        attempted += len(batch)

    mysql_conn.commit()
    cursor.close()
    logging.info("Imported %s/%s rows from %s", inserted, attempted, db_path.relative_to(ROOT_DIR))
    return inserted, attempted


def fetch_destination_count(mysql_conn, schema: str, table: str) -> int:
    cursor = mysql_conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    count = cursor.fetchone()[0]
    cursor.close()
    return int(count)


def main() -> int:
    args = parse_args()
    temp_root = Path(args.temp_root).resolve()
    config_path = Path(args.config).resolve()

    sqlite_dbs = discover_sqlite_databases(temp_root)
    logging.info("Found %s AGC SQLite database(s) under %s", len(sqlite_dbs), temp_root)

    mysql_credentials = load_mysql_credentials(config_path)
    mysql_conn = mysql.connector.connect(**mysql_credentials)

    try:
        ensure_destination(mysql_conn, args.schema)
        inserted_total = 0
        attempted_total = 0

        for db_path in sqlite_dbs:
            rows = read_sqlite_rows(db_path)
            inserted, attempted = import_rows(mysql_conn, args.schema, db_path, rows, args.batch_size)
            inserted_total += inserted
            attempted_total += attempted

        destination_count = fetch_destination_count(mysql_conn, args.schema, args.table)
    finally:
        mysql_conn.close()

    logging.info(
        "Completed AGC import. Attempted=%s Inserted=%s DestinationCount=%s",
        attempted_total,
        inserted_total,
        destination_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
