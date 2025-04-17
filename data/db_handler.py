# data/db_handler.py
import yaml
import logging
import re
import pandas as pd
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime, timedelta
import random
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseHandler:
    def __init__(self, mysql_config_path="mysql/id.yaml", parts_config_path="parts_config/parts.yaml"):
        self.mysql_config_path = mysql_config_path
        self.parts_config_path = parts_config_path
        self.mysqlID = None
        self.mysqlPassword = None
        self.mysqlHost = None
        self.mysqlHostPort = None
        self.config = None

        self.load_mysql_credentials()
        self.load_parts_config()

    def load_mysql_credentials(self):
        """Load MySQL credentials from a YAML file."""
        try:
            with open(self.mysql_config_path, "r") as file:
                credentials = yaml.safe_load(file)
                self.mysqlID = credentials.get("id")
                self.mysqlPassword = credentials.get("pass")
                self.mysqlHost = credentials.get("host")
                self.mysqlHostPort = credentials.get("port")
                logging.info("MySQL credentials loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading MySQL credentials: {e}")

    def load_parts_config(self):
        """Load parts configuration from a YAML file."""
        try:
            with open(self.parts_config_path, "r") as file:
                self.config = yaml.safe_load(file).get("parts", {})
                logging.info("Parts configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading parts config: {e}")

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from MySQL with proper connection handling.
        """
        conn = None
        try:
            conn = mysql.connector.connect(
                user=self.mysqlID,
                password=self.mysqlPassword,
                host=self.mysqlHost,
                port=self.mysqlHostPort,
                database="AIKENSAresults"  # Replace with your database name if different
            )
            if conn.is_connected():
                df = pd.read_sql(query, conn)
                logging.info("Data fetched successfully from MySQL.")
                return df
        except Error as e:
            logging.error(f"Error while connecting to MySQL: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()
        return pd.DataFrame()

    def clean_detected_pitch(self, row: dict):
        """Clean and separate the detected pitch from extra information."""
        part_name = row.get('partName')
        if part_name not in self.config:
            return None, []
        pitch_config = self.config[part_name]
        detected_pitch_raw = row.get('detected_pitch', "")
        detected_pitch_raw = re.sub(r'[\[\]]', '', detected_pitch_raw)
        pitch_values = []
        for x in detected_pitch_raw.split(','):
            x = x.strip()
            if re.match(r'^-?\d+(\.\d+)?$', x):
                try:
                    pitch_values.append(round(float(x), 1))
                except ValueError:
                    continue
        actual_pitch_count = pitch_config['pitch_count'] - pitch_config.get('num_of_extra_info', 0)
        total_expected_count = pitch_config['pitch_count']
        if len(pitch_values) != total_expected_count:
            return None, []
        main_pitch_values = pitch_values[:actual_pitch_count]
        extra_info_values = pitch_values[actual_pitch_count:]
        return main_pitch_values, extra_info_values

    def clean_resultpitch(self, row: dict):
        """Clean the resultpitch data and convert to numeric values."""
        resultpitch_raw = row.get('resultpitch')
        if not resultpitch_raw:
            return []
        resultpitch_raw = re.sub(r'[\[\]]', '', resultpitch_raw)
        resultpitch_values = []
        for x in resultpitch_raw.split(','):
            x = x.strip()
            if re.match(r'^-?\d+(\.\d+)?$', x):
                try:
                    resultpitch_values.append(round(float(x), 1) if '.' in x else int(x))
                except ValueError:
                    continue
        return resultpitch_values

    def clean_numofPart(self, value):
        """Standardize 'numofPart' values."""
        if isinstance(value, str):
            standardized_value = re.sub(r'\((\d+),\s*(\d+)\)', r'[\1, \2]', value)
            try:
                evaluated_value = eval(standardized_value)
                if isinstance(evaluated_value, list) and len(evaluated_value) == 2:
                    return evaluated_value
            except Exception as e:
                logging.error(f"Error evaluating numofPart: {e}")
        return [0, 0]

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the DataFrame for consumption by the Dash app."""
        if data is None or data.empty:
            logging.warning("No data fetched from MySQL.")
            return pd.DataFrame()
        # Clean detected pitch and resultpitch fields
        data[['cleaned_pitch', 'extra_info']] = data.apply(
            lambda row: pd.Series(self.clean_detected_pitch(row)), axis=1
        )
        data['cleaned_resultpitch'] = data.apply(lambda row: self.clean_resultpitch(row), axis=1)
        data = data.dropna(subset=['cleaned_pitch'])
        data['numofPart'] = data['numofPart'].apply(self.clean_numofPart)
        data['currentnumofPart'] = data['currentnumofPart'].apply(self.clean_numofPart)
        max_extra_info_count = data['extra_info'].apply(len).max() if not data.empty else 0
        for i in range(max_extra_info_count):
            data[f'extra_info_{i+1:02}'] = data['extra_info'].apply(lambda x: x[i] if i < len(x) else None)
        data = data.drop(columns=['extra_info'], errors='ignore')
        # Convert list columns to JSON strings
        for col in ['numofPart', 'currentnumofPart', 'detected_pitch', 'delta_pitch', 'cleaned_pitch']:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        data['timestampDate'] = pd.to_datetime(data['timestampDate'], format='%Y%m%d', errors='coerce')
        data['timestampHour'] = pd.to_datetime(data['timestampHour'], format='%H:%M:%S', errors='coerce').dt.time
        data['full_timestamp'] = data.apply(
            lambda row: pd.to_datetime(f"{row['timestampDate'].date()} {row['timestampHour']}") 
            if pd.notnull(row['timestampDate']) and pd.notnull(row['timestampHour']) else None, axis=1
        )
        data = data.sort_values(by=['partName', 'full_timestamp']).reset_index(drop=True)
        data['kensaTime'] = data.groupby('partName')['full_timestamp'].diff().dt.total_seconds().fillna(0)
        data['kensaTime'] = data['kensaTime'].apply(lambda x: 0 if x > 240 else x)
        data = data.drop(columns=['timestampHour', 'timestampDate', 'detected_pitch', 'delta_pitch', 'total_length'], errors='ignore')
        logging.info("Data preprocessing completed successfully.")
        return data

    def load_combined_data(self) -> pd.DataFrame:
        """Fetch, clean, and return the complete data."""
        query = "SELECT * FROM AIKENSAresults.inspection_results"
        raw_data = self.fetch_data(query)
        return self.preprocess_data(raw_data)
