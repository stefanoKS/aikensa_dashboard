from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, ctx, no_update
from dash.dcc import Download
from dash.dcc import send_bytes
import dash_bootstrap_components as dbc
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import yaml
import os
import json
from datetime import datetime, timedelta
import io 
import mysql.connector
from mysql.connector import Error
import logging
import random
import hashlib

def random_color_for_reason(reason):
    """
    Generates a random hex color deterministically based on the reason string.
    This uses a hash of the string to seed a local random generator.
    """
    # Create a hash of the reason using SHA-256
    hash_digest = hashlib.sha256(reason.encode('utf-8')).hexdigest()
    # Convert a portion of the digest to an integer
    seed = int(hash_digest[:8], 16)
    # Create a local Random instance seeded with the hash
    rng = random.Random(seed)
    return '#{:06x}'.format(rng.randint(0, 0xFFFFFF))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseHandler:
    def __init__(self, mysql_config_path="mysql/id.yaml", parts_config_path="./parts_config/parts.yaml"):
        self.mysql_conn = None
        self.mysqlID = None
        self.mysqlPassword = None
        self.mysqlHost = None
        self.mysqlHostPort = None
        self.config = None
        self.combined_df = None

        self.mysql_config_path = mysql_config_path
        self.parts_config_path = parts_config_path

        # Load MySQL credentials and parts configuration
        self.load_mysql_credentials()
        self.load_config()

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

    def load_config(self):
        """Load parts configuration from a YAML file."""
        try:
            with open(self.parts_config_path, "r") as file:
                self.config = yaml.safe_load(file).get("parts", {})
                logging.info("Parts configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading parts config: {e}")

    def fetch_data(self, query):
        """Fetch data from the MySQL database with proper connection handling."""
        conn = None
        try:
            conn = mysql.connector.connect(
                user=self.mysqlID,
                password=self.mysqlPassword,
                host=self.mysqlHost,
                port=self.mysqlHostPort,
                database="AIKENSAresults"  # Specify your database name here
            )
            if conn.is_connected():
                df = pd.read_sql(query, conn)
                logging.info("Data fetched successfully from MySQL.")
                return df
        except Error as e:
            logging.error(f"Error while connecting to MySQL: {e}")
            return None
        finally:
            if conn is not None and conn.is_connected():
                conn.close()

    def clean_detected_pitch(self, row):
        """Clean the detected pitch data based on the part configuration."""
        part_name = row.get('partName')
        if part_name not in self.config:
            return None, []

        pitch_config = self.config[part_name]
        detected_pitch_raw = row.get('detected_pitch', "")
        # Remove square brackets
        detected_pitch_raw = re.sub(r'[\[\]]', '', detected_pitch_raw)

        # Convert valid entries to floats with one decimal
        pitch_values = []
        for x in detected_pitch_raw.split(','):
            x = x.strip()
            if re.match(r'^-?\d+(\.\d+)?$', x):
                try:
                    pitch_values.append(round(float(x), 1))
                except ValueError:
                    continue

        # Determine expected pitch counts
        actual_pitch_count = pitch_config['pitch_count'] - pitch_config.get('num_of_extra_info', 0)
        total_expected_count = pitch_config['pitch_count']

        if len(pitch_values) != total_expected_count:
            return None, []

        # Separate main pitch values and extra info
        main_pitch_values = pitch_values[:actual_pitch_count]
        extra_info_values = pitch_values[actual_pitch_count:]

        return main_pitch_values, extra_info_values

    def clean_resultpitch(self, row):
        """Clean the resultpitch data by removing brackets and converting values to numbers."""
        resultpitch_raw = row.get('resultpitch')
        # Check if the value is None or empty, and return an empty list if so
        if not resultpitch_raw:
            return []
        
        # Remove square brackets
        resultpitch_raw = re.sub(r'[\[\]]', '', resultpitch_raw)
        
        # Convert valid entries to numbers (ints if possible, or floats if needed)
        resultpitch_values = []
        for x in resultpitch_raw.split(','):
            x = x.strip()
            if re.match(r'^-?\d+(\.\d+)?$', x):
                try:
                    # If the number has a decimal point, convert to float; otherwise, int
                    if '.' in x:
                        resultpitch_values.append(round(float(x), 1))
                    else:
                        resultpitch_values.append(int(x))
                except ValueError:
                    continue
        return resultpitch_values


    def clean_numofPart(self, value):
        """Clean and standardize 'numofPart' values."""
        if isinstance(value, str):
            # Replace parentheses with square brackets to make it evaluable as a list
            standardized_value = re.sub(r'\((\d+),\s*(\d+)\)', r'[\1, \2]', value)
            try:
                evaluated_value = eval(standardized_value)
                if isinstance(evaluated_value, list) and len(evaluated_value) == 2:
                    return evaluated_value
            except Exception as e:
                logging.error(f"Error evaluating numofPart: {e}")
        return [0, 0]  # Default fallback value

    def preprocess_data(self, data):
        """Preprocess the fetched data before passing it to the next class."""
        if data is None or data.empty:
            logging.warning("Warning: No data fetched from MySQL.")
            return pd.DataFrame()

        # Apply cleaning functions for pitch and extra info
        data[['cleaned_pitch', 'extra_info']] = data.apply(lambda row: pd.Series(self.clean_detected_pitch(row)), axis=1)
        data['cleaned_resultpitch'] = data.apply(lambda row: self.clean_resultpitch(row), axis=1)

        # Drop rows where pitch cleaning failed
        data = data.dropna(subset=['cleaned_pitch'])

        # Standardize 'numofPart' columns
        data['numofPart'] = data['numofPart'].apply(self.clean_numofPart)
        data['currentnumofPart'] = data['currentnumofPart'].apply(self.clean_numofPart)

        # Expand extra_info into separate columns
        max_extra_info_count = data['extra_info'].apply(len).max() if not data.empty else 0
        for i in range(max_extra_info_count):
            data[f'extra_info_{i+1:02}'] = data['extra_info'].apply(lambda x: x[i] if i < len(x) else None)
        data = data.drop(columns=['extra_info'])

        # Convert list columns to JSON strings for database storage
        list_columns = ['numofPart', 'currentnumofPart', 'detected_pitch', 'delta_pitch', 'cleaned_pitch']
        for col in list_columns:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

        # Parse timestamps
        data['timestampDate'] = pd.to_datetime(data['timestampDate'], format='%Y%m%d', errors='coerce')
        data['timestampHour'] = pd.to_datetime(data['timestampHour'], format='%H:%M:%S', errors='coerce').dt.time

        data['full_timestamp'] = data.apply(
            lambda row: pd.to_datetime(f"{row['timestampDate'].date()} {row['timestampHour']}") 
            if pd.notnull(row['timestampDate']) and pd.notnull(row['timestampHour']) else None,
            axis=1
        )

        # Sort data by part name and full timestamp
        data = data.sort_values(by=['partName', 'full_timestamp']).reset_index(drop=True)

        # Calculate kensaTime (time difference per part) and cap extreme values
        data['kensaTime'] = data.groupby('partName')['full_timestamp'].diff().dt.total_seconds().fillna(0)
        data['kensaTime'] = data['kensaTime'].apply(lambda x: 0 if x > 240 else x)

        # Drop unnecessary columns
        columns_to_drop = ['timestampHour', 'timestampDate', 'detected_pitch', 'delta_pitch', 'total_length']
        data = data.drop(columns=columns_to_drop, errors='ignore')

        logging.info("Data preprocessing completed successfully.")
        return data

    def load_combined_data(self):
        query = "SELECT * FROM AIKENSAresults.inspection_results"  # Replace with your actual query
        raw_data = self.fetch_data(query)
        processed_data = self.preprocess_data(raw_data)
        if not processed_data.empty:
            self.min_date = processed_data['full_timestamp'].min().date() - timedelta(days=1)
            self.max_date = processed_data['full_timestamp'].max().date() + timedelta(days=1)
            self.part_options = [{'label': part, 'value': part} 
                                for part in processed_data['partName'].unique()]
            self.default_start_date = (datetime.today() - timedelta(days=1)).date()
            self.default_end_date = (datetime.today() + timedelta(days=1)).date()

        return processed_data

class MyApp:
    def __init__(self, db_handler):
        self.db_handler = db_handler
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "AI検査"
        self.app._favicon = ("aikensa.png")
        self.combined_df = None  # To store the processed dataframe

        self.last_update = datetime.min
        
        self.min_date = None
        self.max_date = None
        self.part_options = []
        self.default_start_date = None
        self.default_end_date = None
        self.load_combined_data()

        self.last_update = datetime.min


    def load_combined_data(self):

        """Fetch and preprocess data every 5 minutes."""
        now = datetime.now()
        # If the cached data is older than 5 minutes or not available, update it.
        if self.combined_df is None or (now - self.last_update) > timedelta(minutes=15):
            self.combined_df = self.db_handler.load_combined_data()
            if self.combined_df is not None and not self.combined_df.empty:
                self.min_date = self.combined_df['full_timestamp'].min().date() - timedelta(days=1)
                self.max_date = self.combined_df['full_timestamp'].max().date() + timedelta(days=1)
                self.part_options = [{'label': part, 'value': part} 
                                     for part in self.combined_df['partName'].unique()]
                self.default_start_date = (datetime.today() - timedelta(days=1)).date()
                self.default_end_date = (datetime.today() + timedelta(days=1)).date()
            self.last_update = now  # Update the timestamp after refreshing data
        return self.combined_df
        

    def get_layout(self):
        """Return the layout for the Dash application."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("検索フィルター", style={'color': 'white'}),
                    html.Hr(style={'border-color': 'white'}),
                    html.Label("製品選択:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id="part-dropdown",
                        options=self.part_options,
                        placeholder="製品名",
                        style={'margin-bottom': '20px'}
                    ),
                    html.Label("日付フィルター:", style={'color': 'white'}),
                    dcc.DatePickerRange(
                        id="date-picker",
                        start_date=self.default_start_date,
                        end_date=self.default_end_date,
                        display_format="YYYY-MM-DD",
                        min_date_allowed=self.min_date,
                        max_date_allowed=self.max_date,
                        style={'margin-bottom': '15px'}
                    ),
                    html.Label("グラフ/一覧タイプ: ", style={'color': 'white'}),
                    dcc.RadioItems(
                        id='view-selection',
                        options=[
                            {'label': '検査結果一覧', 'value': 'inspection'},
                            {'label': 'OK/NG 品数', 'value': 'ok_ng'},
                            {'label': '検査時間', 'value': 'kensa_time'},
                            {'label': 'ピッチデータの詳細', 'value': 'pitch_average'}
                        ],
                        value='inspection',
                        labelStyle={'display': 'block', 'color': 'white'}
                    ),
                ], width=3, style={
                    'background-color': '#8B0000',  # Dark red
                    'padding': '20px',
                    'position': 'fixed',
                    'height': '100vh'
                }),
                dbc.Col([
                    html.H2("AI検査 ダッシュボード", style={'text-align': 'center', 'margin-top': '20px','display': 'block', 'text-decoration': 'underline','font-weight': 'bold','font-family': 'Arial Black, Gadget, sans-serif'}),
                    html.Img(id="part-image", style={'width': '100%', 'height': 'auto', 'margin-bottom': '20px'}),
                    html.Hr(),
                    html.Div(id="dynamic-content", style={'overflowX': 'auto'})
                ], width={"size": 9, "offset": 3})
            ]),
            dcc.Download(id="download-dataframe-xlsx"),
            dcc.Interval(id='interval-component', interval=10 * 60 * 1000, n_intervals=0)
            ], fluid=True)


    def run(self, host="0.0.0.0", port="8050"):
        self.app.layout = self.get_layout()

        @self.app.callback(Output("part-image", "src"), [Input("part-dropdown", "value")])
        def update_image(selected_part):
            image_path = f"assets/parts_img/{selected_part}.png" if selected_part else "assets/parts_img/not_found.png"
            if not os.path.exists(image_path):
                image_path = "assets/parts_img/not_found.png"
            return f"/{image_path}"

        @self.app.callback(
            Output("dynamic-content", "children"),
            [Input("interval-component", "n_intervals"),
             Input("part-dropdown", "value"),
             Input("date-picker", "start_date"),
             Input("date-picker", "end_date"),
             Input("view-selection", "value")]
        )
        def update_content(n_intervals, selected_part, start_date, end_date, view):
            if not selected_part or not start_date or not end_date:
                return "検索製品名また日付を選択してください。"

            # Refresh the combined data
            self.load_combined_data()

            # Convert date strings to datetime objects for proper filtering
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            filtered_df = self.combined_df[
                (self.combined_df['partName'] == selected_part) &
                (self.combined_df['full_timestamp'] >= start_date_dt) &
                (self.combined_df['full_timestamp'] <= end_date_dt)
            ]

            if filtered_df.empty:
                return html.Div("AIKENSAデータはありません。", style={'color': 'red', 'font-weight': 'bold'})

            if view == 'inspection':
                columns_to_display = ['partName', "status", "NGreason", 'full_timestamp', 'numofPart', 'currentnumofPart', 'kensainName',
                                      'cleaned_pitch', 'kensaTime']

                filtered_df = filtered_df[columns_to_display]
                filtered_df = filtered_df.sort_values(by='full_timestamp', ascending=False)

                custom_columns = [
                        {'name': '製品名', 'id': 'partName'},
                        {'name': '検査結果', 'id': 'status'},
                        {'name': 'NG理由', 'id': 'NGreason'},
                        {'name': '検査実施時間', 'id': 'full_timestamp'},
                        {'name': '本日数検査品数', 'id': 'numofPart'},
                        {'name': '現時検査品数', 'id': 'currentnumofPart'},
                        {'name': '検査員番号', 'id': 'kensainName'},
                        {'name': 'ピッチ結果', 'id': 'cleaned_pitch'},
                        {'name': 'サイクルタイム', 'id': 'kensaTime'}
                    ]
                
                download_button = html.Button(
                    "Excelエクスポート",
                    id="download-excel",
                    n_clicks=0,
                    className="btn btn-primary",
                    style={'margin-bottom': '10px'}
                )
                table = html.Div(
                    dash_table.DataTable(
                        data=filtered_df.to_dict('records'),
                        columns=custom_columns,
                        page_size=25,
                        style_table={'overflowX': 'auto'},
                        style_header={'textAlign': 'center'}, 
                        style_cell={'textAlign': 'center'} 
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                )
                return html.Div([download_button, table])
            

            elif view == 'ok_ng':
                # --- Stacked OK/NG Graph ---
                # Group the data by date (using the date portion of full_timestamp)
                daily_summary = filtered_df.groupby(filtered_df['full_timestamp'].dt.date).last()

                # Extract OK and NG counts from the 'numofPart' column
                ok_counts = daily_summary['numofPart'].apply(
                    lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else 0
                )
                ng_counts = daily_summary['numofPart'].apply(
                    lambda x: json.loads(x)[1] if len(json.loads(x)) > 1 else 0
                )

                # Calculate NG percentage for each day
                percentage_ng = []
                for ok, ng in zip(ok_counts, ng_counts):
                    total = ok + ng
                    perc = (ng / total * 100) if total > 0 else 0
                    percentage_ng.append(f"{perc:.1f}%")

                # Create a stacked bar chart figure
                stacked_fig = go.Figure()
                stacked_fig.add_trace(go.Bar(
                    x=daily_summary.index,
                    y=ok_counts,
                    name='OK Part',
                    marker_color='green'
                ))
                stacked_fig.add_trace(go.Bar(
                    x=daily_summary.index,
                    y=ng_counts,
                    name='NG Part',
                    marker_color='red',
                    text=percentage_ng,
                    textposition='outside'
                ))
                stacked_fig.update_layout(
                    barmode='stack',
                    title="日毎のOK/NG 品数とNG割合",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    showlegend=True
                )

                # --- NG Reason Graph ---
                # Filter the NGreason column: remove NaN, empty, "None", and "null" values
                ng_reason_series = filtered_df['NGreason'].dropna().astype(str)
                ng_reason_series = ng_reason_series[~ng_reason_series.str.strip().isin(["", "None", "null"])]

                # Count the occurrences for each unique NG reason
                ng_reason_counts = ng_reason_series.value_counts()

                # Generate a random color for each unique NG reason
                colors = [random_color_for_reason(reason) for reason in ng_reason_counts.index]

                ng_reason_fig = go.Figure()
                ng_reason_fig.add_trace(go.Bar(
                    x=ng_reason_counts.index,
                    y=ng_reason_counts.values,
                    marker={'color': colors}
                ))
                ng_reason_fig.update_layout(
                    title="NG理由ごとの件数",
                    xaxis_title="NG理由",
                    yaxis_title="NGの数",
                    showlegend=False
                )

                if 'cleaned_resultpitch' in filtered_df.columns:
                    # Determine the maximum number of pitch positions among all rows
                    max_positions = filtered_df['cleaned_resultpitch'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

                    # Initialize a counter list for each pitch position
                    resultpitch_ng_counts = [0] * max_positions

                    # Iterate over each row's cleaned_resultpitch list and count zeros (NG)
                    for pitch_list in filtered_df['cleaned_resultpitch']:
                        if isinstance(pitch_list, list):
                            # If the row is entirely zeros, skip it.
                            if all(value == 0 for value in pitch_list):
                                continue
                            for i, value in enumerate(pitch_list):
                                if value == 0:
                                    resultpitch_ng_counts[i] += 1

                    # Create labels for each pitch position (P1, P2, ...)
                    pitch_labels = [f"P{i+1}" for i in range(max_positions)]

                    resultpitch_fig = go.Figure()
                    resultpitch_fig.add_trace(go.Bar(
                        x=pitch_labels,
                        y=resultpitch_ng_counts,
                        marker_color='orange'
                    ))
                    resultpitch_fig.update_layout(
                        title="各ピッチ位置ごとのNG数",
                        xaxis_title="ピッチ位置",
                        yaxis_title="NGの数"
                    )
                else:
                    resultpitch_fig = go.Figure()

                # Return all three graphs in a vertical stack
                return html.Div([
                    dcc.Graph(figure=stacked_fig),
                    dcc.Graph(figure=ng_reason_fig),
                    dcc.Graph(figure=resultpitch_fig)
                ])

            elif view == 'kensa_time':
                # Create a copy of the filtered data and extract the date for grouping.
                df_temp = filtered_df.copy()
                df_temp['Date'] = df_temp['full_timestamp'].dt.date

                # Create a box plot that shows only the outlier points.
                kensa_box_fig = go.Figure(go.Box(
                    x=df_temp['Date'],
                    y=df_temp['kensaTime'],
                    boxpoints='outliers',  # Only display outlier points (外れ値)
                    marker=dict(color='black'),
                    line=dict(color='blue')
                ))
                kensa_box_fig.update_layout(
                    title="日常検査時間の分布",
                    xaxis_title="日付",
                    yaxis_title="検査時間(秒)",
                    showlegend=False
                )

                return html.Div([dcc.Graph(figure=kensa_box_fig)])

            elif view == 'pitch_average':
                # Retrieve part configuration for the selected part.
                part_config = self.db_handler.config.get(selected_part, {})
                pitch_count = part_config.get("pitch_count", 0)
                nominal_pitch = part_config.get("nominal_pitch", [])
                tolerance = part_config.get("tolerance", [])
                
                # Convert cleaned_pitch from JSON string to list.
                filtered_df['cleaned_pitch'] = filtered_df['cleaned_pitch'].apply(json.loads)
                actual_pitch_count = pitch_count - part_config.get("num_of_extra_info", 0)
                
                # Build a DataFrame from the cleaned pitch data.
                expanded_pitch_df = pd.DataFrame(filtered_df['cleaned_pitch'].tolist(), index=filtered_df['full_timestamp'])
                expanded_pitch_df.columns = [f'Pitch {i+1}' for i in range(actual_pitch_count)]
                
                # Add a Date column extracted from the full_timestamp index.
                expanded_pitch_df['Date'] = expanded_pitch_df.index.date

                # For each pitch position, create a box plot (one graph per pitch)
                pitch_graphs = []
                for i in range(actual_pitch_count):
                    pitch_column = f'Pitch {i+1}'
                    # Subset the data for this pitch and the Date.
                    temp_df = expanded_pitch_df[[pitch_column, 'Date']].copy()
                    # Filter out rows where the pitch value is 0 or 0.0.
                    temp_df = temp_df[temp_df[pitch_column] != 0]
                    
                    # Create a box plot with Plotly Express.
                    fig = px.box(
                        temp_df,
                        x='Date',
                        y=pitch_column,
                        title=f'{pitch_column} Distribution by Day',
                        labels={'Date': '日付', pitch_column: 'ピッチ(mm)'}
                    )
                    
                    # Optionally overlay nominal value and tolerance boundaries if available.
                    nominal = nominal_pitch[i] if i < len(nominal_pitch) else None
                    tol = tolerance[i] if i < len(tolerance) else None
                    if nominal is not None:
                        fig.add_shape(
                            type="line",
                            xref="paper", x0=0, x1=1,
                            yref="y", y0=nominal, y1=nominal,
                            line=dict(color="blue", dash="dash")
                        )
                    if nominal is not None and tol is not None:
                        fig.add_shape(
                            type="line",
                            xref="paper", x0=0, x1=1,
                            yref="y", y0=nominal + tol, y1=nominal + tol,
                            line=dict(color="red", dash="dot")
                        )
                        fig.add_shape(
                            type="line",
                            xref="paper", x0=0, x1=1,
                            yref="y", y0=nominal - tol, y1=nominal - tol,
                            line=dict(color="red", dash="dot")
                        )
                    
                    # Calculate daily mean for this pitch and overlay it as a scatter trace.
                    daily_means = temp_df.groupby('Date')[pitch_column].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=daily_means['Date'],
                        y=daily_means[pitch_column],
                        mode='markers',
                        marker=dict(symbol='x', size=10, color='black'),
                        name='Mean'
                    ))
                    
                    pitch_graphs.append(dcc.Graph(figure=fig))
                                
                return html.Div(pitch_graphs)

            return "Invalid view selected."


        @self.app.callback(
            Output("download-dataframe-xlsx", "data"),
            Input("download-excel", "n_clicks"),
            State("part-dropdown", "value"),
            State("date-picker", "start_date"),
            State("date-picker", "end_date"),
            State("view-selection", "value"),
            prevent_initial_call=True  # Ensure callback only triggers on button click
        )

        def download_excel(n_clicks, selected_part, start_date, end_date, view):
            if n_clicks is None or n_clicks == 0 or view != "inspection":
                return no_update

            # Filter the data for the selected part and date range
            columns_to_display = ['partName', 'numofPart', 'currentnumofPart', 'kensainName',
                                'cleaned_pitch', 'full_timestamp', 'kensaTime']
            df_to_export = self.combined_df[
                (self.combined_df['partName'] == selected_part) &
                (self.combined_df['full_timestamp'] >= start_date) &
                (self.combined_df['full_timestamp'] <= end_date)
            ]
            df_to_export = df_to_export[columns_to_display]

            # Convert full_timestamp to string
            df_to_export['full_timestamp'] = df_to_export['full_timestamp'].astype(str)

            if df_to_export.empty:
                return no_update

            # Write Excel data to a BytesIO buffer
            def to_excel(bytes_io):
                with pd.ExcelWriter(bytes_io, engine="xlsxwriter") as writer:
                    df_to_export.to_excel(writer, index=False, sheet_name="Inspection Results")

            # Send the Excel file for download
            return send_bytes(to_excel, "inspection_results.xlsx")

        self.app.run_server(host=host, port=port)

if __name__ == '__main__':
    db_handler = DatabaseHandler()

    query = "SELECT * FROM AIKENSAresults.inspection_results"
    raw_data = db_handler.fetch_data(query)

    preprocessed_data = db_handler.preprocess_data(raw_data)

    #export to excel
    # preprocessed_data.to_excel("preprocessed_data.xlsx")

    my_app = MyApp(db_handler)
    my_app.run(host="0.0.0.0", port="8050")