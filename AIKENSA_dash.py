# dashboard.py
import os
import json
import yaml
import random
from datetime import datetime, timedelta
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table, no_update, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

import traceback
from data.db_handler import DatabaseHandler
from data.data_manager import DataManager

class DashApp:
    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.data_manager = DataManager(self.db_handler)

        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "AI検査"
        self.app._favicon = ("aikensa.png")
        self.combined_df = None
        self.last_update = datetime.min

        # Initialize filter defaults
        self.min_date = None
        self.max_date = None
        self.part_options = []
        self.maker_options = []
        self.cartype_options_by_maker = {}
        self.part_by_maker = {}
        self.default_start_date = None
        self.default_end_date = None
        self.refresh_data()

        self.app.layout = self.get_layout()
        self.register_callbacks()

    def refresh_data(self):
        """Refresh combined data if more than 1 minutes have passed or if not loaded yet."""
        now = datetime.now()
        print(f"Time now: {now}, Last update: {self.last_update}")

        if self.combined_df is None or (now - self.last_update) > timedelta(minutes=1):
            print("Refreshing data...")
            # self.combined_df = self.db_handler.load_combined_data()
            new_df = self.data_manager.fetch_and_update()

            if not new_df.empty:
                self.combined_df = new_df
                self.combined_df['partName'] = (
                    self.combined_df['partName'].astype(str).str.strip().str.upper()
                )

                min_ts = self.combined_df['full_timestamp'].min().date()
                max_ts = self.combined_df['full_timestamp'].max().date()

                # datepicker bounds
                self.min_date = min_ts - timedelta(days=5)
                self.max_date = max_ts + timedelta(days=5)

                # ✅ default range aligned with data (last 7 days ending at max_ts)
                end_default = max_ts
                start_default = max(min_ts, end_default - timedelta(days=7))
                self.default_start_date = start_default
                self.default_end_date = end_default

                # rebuild product dropdown
                self.part_options = [
                    {'label': p, 'value': p}
                    for p in sorted(self.combined_df['partName'].unique())
                ]
                
                cfg = self.db_handler.config or {}
                parts_present = set(self.combined_df['partName'].unique())

                maker_set = set()
                cartype_by_maker = {}                 # maker -> set(cartypes)
                parts_by_maker_cartype = {}           # (maker, cartype) -> set(parts)

                for part in parts_present:
                    meta = cfg.get(part, {})
                    maker = str(meta.get("maker", "Unknown")).strip()
                    cartype = str(meta.get("cartype", "-")).strip()
                    maker_set.add(maker)
                    cartype_by_maker.setdefault(maker, set()).add(cartype)
                    parts_by_maker_cartype.setdefault((maker, cartype), set()).add(part)

                self.maker_options = [{'label': m, 'value': m} for m in sorted(maker_set)]
                self.cartype_options_by_maker = {m: sorted(list(cts)) for m, cts in cartype_by_maker.items()}
                self.parts_by_maker_cartype = {k: sorted(list(v)) for k, v in parts_by_maker_cartype.items()}

                #print for debugging
                # print(f"Available makers: {self.maker_options}")
                # print(f"Cartypes by maker: {self.cartype_options_by_maker}")
                # print(f"Parts by (maker, cartype): {self.parts_by_maker_cartype}")

            self.last_update = now

    def get_layout(self):
        """Return the Dash layout."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("検索フィルター", style={'color': 'white'}),
                    html.Hr(style={'border-color': 'white'}),
                    html.Label("メーカー選択:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id="maker-dropdown",
                        options=self.maker_options,
                        placeholder="メーカー",
                        style={'margin-bottom': '12px'}
                    ),
                    html.Label("車種選択:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id="cartype-dropdown",
                        options=[],              # filled by callback
                        placeholder="車種",
                        style={'margin-bottom': '12px'}
                    ),
                    html.Label("製品選択:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id="part-dropdown",
                        options=[],              # filled by callback
                        placeholder="製品名",
                        style={'margin-bottom': '20px'}
                    ),
                    html.Label("日付フィルター:", style={'color': 'white'}),
                    dcc.DatePickerRange(id="date-picker", display_format="YYYY-MM-DD"),
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
                    'background-color': '#8B0000',
                    'padding': '20px',
                    'position': 'fixed',
                    'height': '100vh'
                }),
                dbc.Col([
                    html.H2("AI検査 ダッシュボード", style={
                        'text-align': 'center',
                        'margin-top': '20px',
                        'display': 'block',
                        'text-decoration': 'underline',
                        'font-weight': 'bold',
                        'font-family': 'Arial Black, Gadget, sans-serif'
                    }),
                    html.Img(id="part-image", style={'width': '100%', 'height': 'auto', 'margin-bottom': '20px'}),
                    html.Hr(),
                    html.Div(id="dynamic-content", style={'overflowX': 'auto'})
                ], width={"size": 9, "offset": 3})
            ]),
            dcc.Download(id="download-dataframe-xlsx"),
            dcc.Interval(id='interval-component', interval=10 * 60 * 1000, n_intervals=0)
        ], fluid=True)

    def register_callbacks(self):

        # --- helpers ---
        def _maps_from_yaml():
            cfg = self.db_handler.config or {}
            maker_set = set()
            cartype_by_maker = {}
            parts_by_maker_cartype = {}
            for part_raw, meta in cfg.items():
                part = str(part_raw).strip().upper()
                maker = str(meta.get("maker", "Unknown")).strip()
                cartype = str(meta.get("cartype", "-")).strip()
                maker_set.add(maker)
                cartype_by_maker.setdefault(maker, set()).add(cartype)
                parts_by_maker_cartype.setdefault((maker, cartype), set()).add(part)
            maker_all = sorted(maker_set)
            cartype_by_maker = {m: sorted(list(s)) for m, s in cartype_by_maker.items()}
            parts_by_maker_cartype = {k: sorted(list(s)) for k, s in parts_by_maker_cartype.items()}
            return maker_all, cartype_by_maker, parts_by_maker_cartype

        def _availability(start_date, end_date):
            """Availability based on current data within date window."""
            self.refresh_data()
            if self.combined_df is None or self.combined_df.empty:
                return set(), {}, {}
            sd = pd.to_datetime(start_date) if start_date else pd.to_datetime(self.default_start_date)
            ed = pd.to_datetime(end_date) if end_date else pd.to_datetime(self.default_end_date)
            ed = ed + pd.Timedelta(hours=23, minutes=59, seconds=59)
            df = self.combined_df
            if 'partName_norm' not in df.columns:
                self.combined_df['partName_norm'] = df['partName'].astype(str).str.strip().str.upper()
            w = self.combined_df[(self.combined_df['full_timestamp'] >= sd) & (self.combined_df['full_timestamp'] <= ed)]
            if w.empty:
                return set(), {}, {}
            parts = set(w['partName_norm'].unique())
            # map via YAML
            _, cartype_all_by_maker, parts_all_by_mc = _maps_from_yaml()
            # invert parts_all_by_mc to part -> (maker, cartype)
            part_to_mc = {}
            for (mk, ct), plist in parts_all_by_mc.items():
                for p in plist:
                    part_to_mc[p] = (mk, ct)
            makers = set()
            cartypes_by_maker = {}
            parts_by_mc = {}
            for p in parts:
                mk, ct = part_to_mc.get(p, ("Unknown", "-"))
                makers.add(mk)
                cartypes_by_maker.setdefault(mk, set()).add(ct)
                parts_by_mc.setdefault((mk, ct), set()).add(p)
            return (
                makers,
                {m: sorted(list(s)) for m, s in cartypes_by_maker.items()},
                {k: sorted(list(s)) for k, s in parts_by_mc.items()},
            )

        # ---------- Datepicker bounds (unchanged) ----------
        @self.app.callback(
            Output("date-picker", "min_date_allowed"),
            Output("date-picker", "max_date_allowed"),
            Output("date-picker", "start_date"),
            Output("date-picker", "end_date"),
            Input("interval-component", "n_intervals"),
        )
        def _refresh_datepicker(_n):
            self.refresh_data()
            return self.min_date, self.max_date, self.default_start_date, self.default_end_date

        # ---------- (1) Maker options (no value here) ----------
        @self.app.callback(
            Output("maker-dropdown", "options"),
            Input("interval-component", "n_intervals"),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
        )
        def _maker_options(_n, start_date, end_date):
            maker_all, _, _ = _maps_from_yaml()
            makers_now, _, _ = _availability(start_date, end_date)
            # Show only available makers; if none in range, fall back to all
            makers = sorted(list(makers_now)) if makers_now else maker_all
            return [{'label': m, 'value': m} for m in makers]

        # ---------- (2) Cartype options + value (triggered by maker changes) ----------
        @self.app.callback(
            Output("cartype-dropdown", "options"),
            Output("cartype-dropdown", "value"),
            Input("maker-dropdown", "value"),
            Input("interval-component", "n_intervals"),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
            State("cartype-dropdown", "value"),
        )
        def _cartype_options_value(maker, _n, start_date, end_date, cur_cartype):
            if not maker:
                return [], None
            _, cartype_all_by_maker, _ = _maps_from_yaml()
            makers_now, cartypes_now_by_maker, _ = _availability(start_date, end_date)
            # limit to available cartypes for this maker; if none, fall back to all YAML cartypes for maker
            cartypes = cartypes_now_by_maker.get(maker, []) or cartype_all_by_maker.get(maker, [])
            opts = [{'label': c, 'value': c} for c in cartypes]
            val = cur_cartype if cur_cartype in cartypes else (cartypes[0] if cartypes else None)
            return opts, val

        # ---------- (3) Part options + value (triggered by maker/cartype changes) ----------
        @self.app.callback(
            Output("part-dropdown", "options"),
            Output("part-dropdown", "value"),
            Input("maker-dropdown", "value"),
            Input("cartype-dropdown", "value"),
            Input("interval-component", "n_intervals"),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
            State("part-dropdown", "value"),
        )
        def _part_options_value(maker, cartype, _n, start_date, end_date, cur_part):
            if not maker or not cartype:
                return [], None
            _, __, parts_all_by_mc = _maps_from_yaml()
            _, ____, parts_now_by_mc = _availability(start_date, end_date)
            parts = parts_now_by_mc.get((maker, cartype), []) or parts_all_by_mc.get((maker, cartype), [])
            opts = [{'label': p, 'value': p} for p in parts]
            val = cur_part if cur_part in parts else (parts[0] if parts else None)
            return opts, val
        

        # ---------- image ----------
        @self.app.callback(Output("part-image", "src"),
                        [Input("maker-dropdown", "value"),
                            Input("cartype-dropdown", "value"),
                            Input("part-dropdown", "value")])
        def update_image(_maker, _cartype, selected_part):
            image_path = f"assets/parts_img/{selected_part}.png" if selected_part else "assets/parts_img/not_found.png"
            if not os.path.exists(image_path):
                image_path = "assets/parts_img/not_found.png"
            return f"/{image_path}"
        

        # ---------- Main dynamic content (inclusive end date) ----------
        @self.app.callback(
            Output("dynamic-content", "children"),
            [
                Input("interval-component", "n_intervals"),
                Input("maker-dropdown", "value"),      
                Input("cartype-dropdown", "value"),     
                Input("part-dropdown", "value"),
                Input("date-picker", "start_date"),
                Input("date-picker", "end_date"),
                Input("view-selection", "value"),
            ]
        )

        def update_content(n_intervals, maker, cartype, selected_part, start_date, end_date, view):
            # you can ignore maker/cartype, but they must be in the signature
            if not selected_part or not start_date or not end_date:
                return "検索製品名また日付を選択してください。"

            self.refresh_data()
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date) + pd.to_timedelta(23, 'h') + pd.to_timedelta(59, 'm') + pd.to_timedelta(59, 's')

            filtered_df = self.combined_df[
                (self.combined_df['partName'] == selected_part) &
                (self.combined_df['full_timestamp'] >= start_date_dt) &
                (self.combined_df['full_timestamp'] <= end_date_dt)
            ]
            if filtered_df.empty:
                return html.Div("AIKENSAデータはありません。", style={'color': 'red', 'font-weight': 'bold'})


            # Handle the different view options (inspection, ok_ng, kensa_time, pitch_average)
            if view == 'inspection':
                columns_to_display = ['partName', "status", "NGreason", 'full_timestamp',
                                    'numofPart', 'currentnumofPart', 'kensainName',
                                    'cleaned_pitch', 'kensaTime', "PPMS"]
                # keep only existing columns (older rows safety)
                columns_to_display = [c for c in columns_to_display if c in filtered_df.columns]
                filtered_df = filtered_df[columns_to_display].sort_values(by='full_timestamp', ascending=False)
                custom_columns = [
                    {'name': '製品名', 'id': 'partName'},
                    {'name': '検査結果', 'id': 'status'},
                    {'name': 'NG理由', 'id': 'NGreason'},
                    {'name': '検査実施時間', 'id': 'full_timestamp'},
                    {'name': '本日数検査品数', 'id': 'numofPart'},
                    {'name': '現時検査品数', 'id': 'currentnumofPart'},
                    {'name': '検査員番号', 'id': 'kensainName'},
                    {'name': 'ピッチ結果', 'id': 'cleaned_pitch'},
                    {'name': 'サイクルタイム', 'id': 'kensaTime'},
                    {'name': 'PPMS番号', 'id': 'PPMS'}
                ]
                custom_columns = [col for col in custom_columns if col['id'] in filtered_df.columns]

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
                daily_summary = filtered_df.groupby(filtered_df['full_timestamp'].dt.date).last()
                ok_counts = daily_summary['numofPart'].apply(lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else 0)
                ng_counts = daily_summary['numofPart'].apply(lambda x: json.loads(x)[1] if len(json.loads(x)) > 1 else 0)
                percentage_ng = [f"{(ng/(ok+ng)*100):.1f}%" if (ok+ng) > 0 else "0%"
                                for ok, ng in zip(ok_counts, ng_counts)]

                stacked_fig = go.Figure()
                stacked_fig.add_trace(go.Bar(
                    x=daily_summary.index,
                    y=ok_counts,
                    name='OK数',
                    marker_color='green'
                ))
                stacked_fig.add_trace(go.Bar(
                    x=daily_summary.index,
                    y=ng_counts,
                    name='NG数',
                    marker_color='red',
                    text=percentage_ng,
                    textposition='outside'
                ))
                stacked_fig.update_layout(
                    barmode='stack',
                    title="日毎のOK/NG 品数とNG割合",
                    xaxis_title="日付",
                    yaxis_title="数",
                    showlegend=True
                )

                ng_reason_series = filtered_df['NGreason'].dropna().astype(str)
                ng_reason_series = ng_reason_series[~ng_reason_series.str.strip().isin(["", "None", "null"])]
                ng_reason_counts = ng_reason_series.value_counts().sort_values(ascending=False)
                reasons = list(ng_reason_counts.index)
                counts = list(ng_reason_counts.values)

                try:
                    with open("./yaml/translation.yaml", "r", encoding="utf-8") as file:
                        translation = yaml.safe_load(file)
                except Exception as e:
                    print("Error loading translation.yaml:", e)
                    translation = {}

                translated_reasons = [translation.get(reason, reason) for reason in reasons]

                import numpy as np
                cumulative = np.cumsum(counts)
                total = np.sum(counts) if len(counts) else 1
                cumulative_percent = (cumulative / total) * 100

                x_bar = list(range(1, len(translated_reasons) + 1))
                x_line = [0] + x_bar
                cumulative_line = np.insert(cumulative_percent, 0, 0)
                text_line = [f"{val:.1f}%" for val in cumulative_line]

                from plotly.subplots import make_subplots
                pareto_fig = make_subplots(specs=[[{"secondary_y": True}]])
                pareto_fig.add_trace(
                    go.Bar(x=x_bar, y=counts, name="NG 数", marker_color="red"),
                    secondary_y=False
                )
                pareto_fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=cumulative_line,
                        text=text_line,
                        mode="lines+markers+text",
                        textposition="top center",
                        name="累積比率 %",
                        marker=dict(color="darkblue")
                    ),
                    secondary_y=True
                )
                pareto_fig.update_layout(
                    title="パレート図",
                    xaxis_title="NG項目",
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=x_bar,
                        ticktext=translated_reasons
                    )
                )
                pareto_fig.update_yaxes(title_text="数", secondary_y=False)
                pareto_fig.update_yaxes(title_text="累積比率 %", secondary_y=True, range=[0, 110])

                if 'cleaned_resultpitch' in filtered_df.columns:
                    filtered_df['cleaned_resultpitch'] = (
                        filtered_df['cleaned_resultpitch']
                        .apply(lambda x: x.tolist() if isinstance(x, np.ndarray)
                                            else x if isinstance(x, list)
                                            else [])
                    )
                    max_positions = filtered_df['cleaned_resultpitch'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
                    resultpitch_ng_counts = [0] * max_positions
                    for pitch_list in filtered_df['cleaned_resultpitch']:
                        if isinstance(pitch_list, list):
                            if all(value == 0 for value in pitch_list):
                                continue
                            for i, value in enumerate(pitch_list):
                                if value == 0:
                                    resultpitch_ng_counts[i] += 1
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

                return html.Div([
                    dcc.Graph(figure=stacked_fig),
                    dcc.Graph(figure=pareto_fig),
                    dcc.Graph(figure=resultpitch_fig)
                ])

            elif view == 'kensa_time':
                df_temp = filtered_df.copy()
                df_temp['Date'] = df_temp['full_timestamp'].dt.date
                kensa_box_fig = go.Figure(go.Box(x=df_temp['Date'], y=df_temp['kensaTime'],
                                                boxpoints='outliers', marker=dict(color='black'), line=dict(color='blue')))
                kensa_box_fig.update_layout(title="日常検査時間の分布", xaxis_title="日付", yaxis_title="検査時間(秒)", showlegend=False)
                return html.Div([dcc.Graph(figure=kensa_box_fig)])

            elif view == 'pitch_average':
                part_config = self.db_handler.config.get(selected_part, {})
                pitch_count = part_config.get("pitch_count", 0)
                nominal_pitch = part_config.get("nominal_pitch", [])
                tolerance = part_config.get("tolerance", [])
                filtered_df = filtered_df.copy()
                filtered_df['cleaned_pitch'] = filtered_df['cleaned_pitch'].apply(json.loads)
                actual_pitch_count = pitch_count - part_config.get("num_of_extra_info", 0)
                expanded_pitch_df = pd.DataFrame(filtered_df['cleaned_pitch'].tolist(), index=filtered_df['full_timestamp'])
                # protect from column mismatch
                if expanded_pitch_df.shape[1] < actual_pitch_count:
                    for pad_i in range(expanded_pitch_df.shape[1], actual_pitch_count):
                        expanded_pitch_df[pad_i] = 0
                    expanded_pitch_df = expanded_pitch_df.iloc[:, :actual_pitch_count]
                expanded_pitch_df.columns = [f'Pitch {i+1}' for i in range(actual_pitch_count)]
                expanded_pitch_df['Date'] = expanded_pitch_df.index.date
                pitch_graphs = []
                for i in range(actual_pitch_count):
                    pitch_column = f'Pitch {i+1}'
                    temp_df = expanded_pitch_df[[pitch_column, 'Date']].copy()
                    temp_df = temp_df[temp_df[pitch_column] != 0]
                    fig = px.box(temp_df, x='Date', y=pitch_column, title=f'{pitch_column} Distribution by Day',
                                labels={'Date': '日付', pitch_column: 'ピッチ(mm)'})
                    nominal = nominal_pitch[i] if i < len(nominal_pitch) else None
                    tol = tolerance[i] if i < len(tolerance) else None
                    if nominal is not None:
                        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=nominal, y1=nominal,
                                    line=dict(color="blue", dash="dash"))
                    if nominal is not None and tol is not None:
                        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=nominal + tol, y1=nominal + tol,
                                    line=dict(color="red", dash="dot"))
                        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=nominal - tol, y1=nominal - tol,
                                    line=dict(color="red", dash="dot"))
                    daily_means = temp_df.groupby('Date')[pitch_column].mean().reset_index()
                    fig.add_trace(go.Scatter(x=daily_means['Date'], y=daily_means[pitch_column],
                                            mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Mean'))
                    pitch_graphs.append(dcc.Graph(figure=fig))
                return html.Div(pitch_graphs)

            return "Invalid view selected."

        # ---------- Excel export (uses inclusive end date) ----------
        @self.app.callback(
            Output("download-dataframe-xlsx", "data"),
            Input("download-excel", "n_clicks"),
            State("part-dropdown", "value"),
            State("date-picker", "start_date"),
            State("date-picker", "end_date"),
            State("view-selection", "value"),
            prevent_initial_call=True
        )
        def download_excel(n_clicks, selected_part, start_date, end_date, view):
            if not n_clicks or view != "inspection":
                return no_update
            if not selected_part or not start_date or not end_date:
                return no_update

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.to_timedelta(23, unit='h') + pd.to_timedelta(59, unit='m') + pd.to_timedelta(59, unit='s')

            columns_to_display = ['partName', 'numofPart', 'currentnumofPart', 'kensainName',
                                'cleaned_pitch', 'full_timestamp', 'kensaTime', "PPMS"]
            df_to_export = self.combined_df[
                (self.combined_df['partName'] == selected_part) &
                (self.combined_df['full_timestamp'] >= start_dt) &
                (self.combined_df['full_timestamp'] <= end_dt)
            ].copy()

            cols_present = [c for c in columns_to_display if c in df_to_export.columns]
            df_to_export = df_to_export[cols_present]
            if 'full_timestamp' in df_to_export.columns:
                df_to_export['full_timestamp'] = df_to_export['full_timestamp'].astype(str)

            if df_to_export.empty:
                return no_update

            def to_excel(bytes_io):
                with pd.ExcelWriter(bytes_io, engine="xlsxwriter") as writer:
                    df_to_export.to_excel(writer, index=False, sheet_name="Inspection Results")

            return dcc.send_bytes(to_excel, "inspection_results.xlsx")

       

    def run(self, host="0.0.0.0", port=8050):
        self.app.run_server(host=host, port=port)

if __name__ == '__main__':
    app_instance = DashApp()
    app_instance.run()
