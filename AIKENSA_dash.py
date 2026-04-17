# dashboard.py
import os
import json
import yaml
import random
from datetime import datetime, timedelta
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table, no_update, Output, Input, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

import traceback
import mysql.connector
from data.db_handler import DatabaseHandler
from data.data_manager import DataManager
from data.agc_lot_data import AgcLotDataProcessor
from agc_lot_visualization import build_agc_lot_content
from spare_data import pushdata

class DashApp:
    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.data_manager = DataManager(self.db_handler)
        self.agc_processor = AgcLotDataProcessor()
        self.agc_maker = "AGC"
        self.agc_cartype = "AGC LOT"
        self.agc_view = "agc_lot"
        self.agc_all_parts_value = "ALL"

        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "AI検査"
        self.app._favicon = ("aikensa.png")
        self.combined_df = None
        self.last_update = datetime.min
        self.refresh_interval_minutes = 1
        self.refresh_interval_ms = self.refresh_interval_minutes * 60 * 1000
        self.latest_data_timestamp = None
        self.last_agc_sync = datetime.min
        self.last_agc_sync_stats = {"attempted": 0, "inserted": 0, "destination_count": 0}
        self.last_agc_sync_error = None

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

    def _get_maker_options_with_agc(self, options=None):
        option_list = list(options if options is not None else self.maker_options)
        option_values = {option.get('value') for option in option_list}
        if self.agc_maker not in option_values:
            option_list.append({'label': self.agc_maker, 'value': self.agc_maker})
        return sorted(option_list, key=lambda option: str(option.get('label', '')))

    def _get_date_context(self, maker=None):
        if maker == self.agc_maker:
            agc_context = self.agc_processor.get_date_context()
            if agc_context:
                return agc_context
        return {
            'min_date': self.min_date,
            'max_date': self.max_date,
            'default_start_date': self.default_start_date,
            'default_end_date': self.default_end_date,
        }

    def _get_agc_part_label(self, selected_part):
        if selected_part in (None, '', self.agc_all_parts_value):
            return '全て'
        return self.agc_processor.get_part_label(selected_part)

    def _get_agc_lot_data(self, selected_part, start_date, end_date):
        part_filter = None if selected_part in (None, '', self.agc_all_parts_value) else str(selected_part)
        return self.agc_processor.get_lot_summary(part_filter=part_filter, start_date=start_date, end_date=end_date)

    def _build_agc_lot_export_df(self, selected_part, start_date, end_date):
        _, display_summary, _, _, _ = self._get_agc_lot_data(selected_part, start_date, end_date)
        if display_summary.empty:
            return display_summary

        export_columns = {
            'partLabel': '品番',
            'lotNumber': 'ロット番号',
            'ok_total': 'OK',
            'ng_total': 'NG',
            'total_parts': '総部品数',
            'parts_per_hour': '時間当たり(本/時間)',
            'sec_per_part': '秒/部品',
            'first_time': '開始時刻',
            'last_time': '終了時刻',
            'duration_min': '実作業時間(分)',
            'time_to_next_lot_min': '次ロットまで(分)',
        }
        return display_summary.rename(columns=export_columns)[list(export_columns.values())]

    def _expand_pitch_result_columns(self, df, selected_part):
        if df.empty or 'cleaned_pitch' not in df.columns:
            return df, []

        part_key = str(selected_part).strip().upper() if selected_part is not None else None
        part_config = (self.db_handler.config or {}).get(part_key, {})
        configured_pitch_count = int(part_config.get('pitch_count', 0) or 0)
        extra_info_count = int(part_config.get('num_of_extra_info', 0) or 0)
        expected_pitch_count = max(configured_pitch_count - extra_info_count, 0)

        pitch_values = df['cleaned_pitch'].apply(
            lambda value: json.loads(value) if isinstance(value, str) and value else value if isinstance(value, list) else []
        )
        detected_pitch_count = max((len(values) for values in pitch_values), default=0)
        pitch_count = expected_pitch_count or detected_pitch_count
        if pitch_count <= 0:
            return df.drop(columns=['cleaned_pitch']), []

        pitch_columns = [f'P{i + 1}' for i in range(pitch_count)]
        pitch_df = pd.DataFrame(
            pitch_values.apply(lambda values: list(values[:pitch_count]) + [None] * max(pitch_count - len(values), 0)).tolist(),
            columns=pitch_columns,
            index=df.index,
        )
        expanded_df = pd.concat([df.drop(columns=['cleaned_pitch']), pitch_df], axis=1)
        return expanded_df, pitch_columns

    def _build_kpi_card(self, title, value, accent_color, subtitle=None):
        body_children = [
            html.Div(title, style={
                'font-size': '0.82rem',
                'font-weight': '600',
                'color': '#6b7280',
                'letter-spacing': '0.04em'
            }),
            html.Div(value, style={
                'font-size': '1.7rem',
                'font-weight': '700',
                'color': '#111827',
                'line-height': '1.1'
            })
        ]
        if subtitle:
            body_children.append(html.Div(subtitle, style={
                'font-size': '0.84rem',
                'color': '#4b5563',
                'margin-top': '6px'
            }))

        return dbc.Card(
            dbc.CardBody(body_children),
            style={
                'border': f'1px solid {accent_color}',
                'border-left': f'6px solid {accent_color}',
                'border-radius': '12px',
                'box-shadow': '0 6px 18px rgba(15, 23, 42, 0.08)',
                'height': '100%',
                'background-color': '#ffffff'
            }
        )

    def _build_filter_badge(self, label, value, color='secondary'):
        return html.Div([
            html.Span(f"{label}: ", style={'font-weight': '600', 'margin-right': '4px'}),
            dbc.Badge(value or '未選択', color=color, className='me-2')
        ], style={'display': 'inline-block', 'margin-right': '10px', 'margin-bottom': '8px'})

    def _build_empty_state(self, title, description, hints=None):
        hints = hints or []
        hint_items = [html.Li(hint, style={'margin-bottom': '6px'}) for hint in hints]
        return dbc.Card(
            dbc.CardBody([
                html.H4(title, style={'font-weight': '700', 'color': '#991b1b'}),
                html.P(description, style={'color': '#374151', 'margin-bottom': '14px'}),
                html.Ul(hint_items, style={'color': '#4b5563', 'padding-left': '20px', 'margin-bottom': '0'}) if hint_items else html.Div()
            ]),
            style={
                'border': '1px dashed #d97706',
                'border-radius': '14px',
                'background-color': '#fff7ed',
                'box-shadow': '0 8px 20px rgba(217, 119, 6, 0.08)'
            }
        )

    def _format_timestamp(self, value):
        if value in (None, datetime.min):
            return '未取得'
        timestamp = pd.to_datetime(value, errors='coerce')
        if pd.isna(timestamp):
            return '未取得'
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def _build_refresh_chip(self, label, value, accent_color='#1d4ed8'):
        return html.Div([
            html.Div(label, style={'font-size': '0.76rem', 'font-weight': '700', 'color': '#6b7280', 'margin-bottom': '4px'}),
            html.Div(value, style={'font-size': '0.95rem', 'font-weight': '700', 'color': '#111827'})
        ], style={
            'padding': '10px 14px',
            'border': f'1px solid {accent_color}',
            'border-radius': '12px',
            'background-color': '#ffffff',
            'min-width': '180px'
        })

    def _sync_agc_data(self):
        attempted_total = 0
        inserted_total = 0
        destination_count = self.last_agc_sync_stats.get('destination_count', 0)

        try:
            sqlite_dbs = pushdata.discover_sqlite_databases(pushdata.DEFAULT_TEMP_ROOT)
            mysql_credentials = pushdata.load_mysql_credentials(pushdata.DEFAULT_CONFIG_PATH)
            mysql_conn = mysql.connector.connect(**mysql_credentials)
            try:
                pushdata.ensure_destination(mysql_conn, pushdata.DEFAULT_SCHEMA)
                for db_path in sqlite_dbs:
                    rows = pushdata.read_sqlite_rows(db_path)
                    inserted, attempted = pushdata.import_rows(
                        mysql_conn,
                        pushdata.DEFAULT_SCHEMA,
                        db_path,
                        rows,
                        batch_size=1000,
                    )
                    inserted_total += inserted
                    attempted_total += attempted
                destination_count = pushdata.fetch_destination_count(
                    mysql_conn,
                    pushdata.DEFAULT_SCHEMA,
                    pushdata.DEFAULT_TABLE,
                )
            finally:
                mysql_conn.close()

            self.last_agc_sync = datetime.now()
            self.last_agc_sync_stats = {
                'attempted': attempted_total,
                'inserted': inserted_total,
                'destination_count': destination_count,
            }
            self.last_agc_sync_error = None
        except Exception as exc:
            logging.exception('AGC sync failed during dashboard refresh')
            self.last_agc_sync_error = str(exc)

    def _build_refresh_status_bar(self):
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div('データ更新ステータス', style={'font-size': '1rem', 'font-weight': '700', 'color': '#111827', 'margin-bottom': '12px'}),
                        html.Div([
                            self._build_refresh_chip('自動更新', f'{self.refresh_interval_minutes} 分ごと', '#1d4ed8'),
                            self._build_refresh_chip('ダッシュボード最終更新', self._format_timestamp(self.last_update), '#15803d'),
                        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'})
                    ], md=10),
                    dbc.Col([
                        dbc.Button(
                            '今すぐ更新',
                            id='manual-refresh-button',
                            n_clicks=0,
                            color='danger',
                            className='w-100',
                            style={'font-weight': '700', 'margin-top': '28px'}
                        )
                    ], md=2)
                ], className='g-2 align-items-start')
            ]),
            style={
                'border': '1px solid #d1d5db',
                'border-radius': '16px',
                'box-shadow': '0 8px 20px rgba(15, 23, 42, 0.08)',
                'background-color': '#f8fafc',
                'margin-top': '20px',
                'margin-bottom': '20px',
            }
        )

    def refresh_data(self, force=False):
        """Refresh AIKENSA data and sync AGC data when the refresh interval elapses or a manual refresh is requested."""
        now = datetime.now()
        should_refresh = force or self.combined_df is None or (now - self.last_update) > timedelta(minutes=self.refresh_interval_minutes)
        if not should_refresh:
            return

        logging.info("Refreshing dashboard data. force=%s last_update=%s", force, self.last_update)
        self._sync_agc_data()

        try:
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
                self.latest_data_timestamp = self.combined_df['full_timestamp'].max()
            elif self.combined_df is None:
                self.latest_data_timestamp = None
        finally:
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
                        options=self._get_maker_options_with_agc(),
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
                    dcc.RadioItems(
                        id='date-preset',
                        options=[
                            {'label': '本日', 'value': 'today'},
                            {'label': '7日', 'value': 'last_7_days'},
                            {'label': '30日', 'value': 'last_30_days'},
                            {'label': 'カスタム', 'value': 'custom'}
                        ],
                        value='last_7_days',
                        labelStyle={'display': 'block', 'color': 'white', 'margin-bottom': '4px'},
                        style={'margin-bottom': '10px'}
                    ),
                    dcc.DatePickerRange(id="date-picker", display_format="YYYY-MM-DD", minimum_nights=0),
                    html.Label("グラフ/一覧タイプ: ", style={'color': 'white'}),
                    dcc.RadioItems(
                        id='view-selection',
                        options=[
                            {'label': '検査結果一覧', 'value': 'inspection'},
                            {'label': 'OK/NG 品数', 'value': 'ok_ng'},
                            {'label': '検査時間', 'value': 'kensa_time'},
                            {'label': 'ピッチデータの詳細', 'value': 'pitch_average'},
                            {'label': 'AGC ロット分析', 'value': 'agc_lot'}
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
                    html.Div(id='refresh-status-bar', children=self._build_refresh_status_bar()),
                    html.Div(id="summary-content", style={'margin-top': '20px', 'margin-bottom': '20px'}),
                    html.Img(id="part-image", style={'width': '100%', 'height': 'auto', 'margin-bottom': '20px'}),
                    html.Hr(),
                    html.Div(id="dynamic-content", style={'overflowX': 'auto'})
                ], width={"size": 9, "offset": 3})
            ]),
            dcc.Download(id="download-dataframe-xlsx"),
            dcc.Store(id='refresh-token', data=self._format_timestamp(self.last_update)),
            dcc.Interval(id='interval-component', interval=self.refresh_interval_ms, n_intervals=0)
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

        @self.app.callback(
            Output('refresh-token', 'data'),
            Output('refresh-status-bar', 'children'),
            Input('interval-component', 'n_intervals'),
            Input('manual-refresh-button', 'n_clicks'),
        )
        def _refresh_dashboard_state(_n_intervals, _manual_clicks):
            self.refresh_data(force=ctx.triggered_id == 'manual-refresh-button')
            return datetime.now().isoformat(), self._build_refresh_status_bar()

        # ---------- Datepicker bounds (unchanged) ----------
        @self.app.callback(
            Output("date-picker", "min_date_allowed"),
            Output("date-picker", "max_date_allowed"),
            Output("date-picker", "start_date"),
            Output("date-picker", "end_date"),
            Input('refresh-token', 'data'),
            Input("maker-dropdown", "value"),
            Input("date-preset", "value"),
            State("date-picker", "start_date"),
            State("date-picker", "end_date"),
        )
        def _refresh_datepicker(_refresh_token, maker, preset_value, current_start_date, current_end_date):
            self.refresh_data()

            date_context = self._get_date_context(maker)
            end_default = date_context.get('default_end_date')
            min_date = date_context.get('min_date')
            max_date = date_context.get('max_date')

            def _resolve_preset_dates(preset):
                if end_default is None:
                    return None, None
                if preset == 'today':
                    return end_default, end_default
                if preset == 'last_30_days':
                    return max(min_date, end_default - timedelta(days=29)), end_default
                if preset == 'custom':
                    return current_start_date, current_end_date
                return max(min_date, end_default - timedelta(days=6)), end_default

            trigger = ctx.triggered_id
            next_start_date, next_end_date = current_start_date, current_end_date

            if trigger in {'date-preset', 'maker-dropdown'} or not current_start_date or not current_end_date:
                next_start_date, next_end_date = _resolve_preset_dates(preset_value)

            if next_start_date and next_end_date:
                next_start_date = pd.to_datetime(next_start_date).date()
                next_end_date = pd.to_datetime(next_end_date).date()

                if min_date and next_start_date < min_date:
                    next_start_date = min_date
                if max_date and next_end_date > max_date:
                    next_end_date = max_date
                if next_start_date > next_end_date:
                    next_start_date = next_end_date

            return min_date, max_date, next_start_date, next_end_date

        # ---------- (1) Maker options (no value here) ----------
        @self.app.callback(
            Output("maker-dropdown", "options"),
            Input('refresh-token', 'data'),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
        )
        def _maker_options(_refresh_token, start_date, end_date):
            maker_all, _, _ = _maps_from_yaml()
            makers_now, _, _ = _availability(start_date, end_date)
            # Show only available makers; if none in range, fall back to all
            makers = sorted(list(makers_now)) if makers_now else maker_all
            return self._get_maker_options_with_agc([{'label': m, 'value': m} for m in makers])

        # ---------- (2) Cartype options + value (triggered by maker changes) ----------
        @self.app.callback(
            Output("cartype-dropdown", "options"),
            Output("cartype-dropdown", "value"),
            Input("maker-dropdown", "value"),
            Input('refresh-token', 'data'),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
            State("cartype-dropdown", "value"),
        )
        def _cartype_options_value(maker, _refresh_token, start_date, end_date, cur_cartype):
            if maker == self.agc_maker:
                agc_options = [{'label': self.agc_cartype, 'value': self.agc_cartype}]
                next_value = cur_cartype if cur_cartype == self.agc_cartype else self.agc_cartype
                return agc_options, next_value
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
            Input('refresh-token', 'data'),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
            State("part-dropdown", "value"),
        )
        def _part_options_value(maker, cartype, _refresh_token, start_date, end_date, cur_part):
            if maker == self.agc_maker:
                parts = self.agc_processor.get_available_parts(start_date=start_date, end_date=end_date)
                if not parts:
                    parts = self.agc_processor.get_available_parts()
                parts = self.agc_processor.get_selectable_parts(parts)

                options = [{'label': '全て', 'value': self.agc_all_parts_value}] + [
                    {'label': self.agc_processor.get_part_label(part), 'value': part} for part in parts
                ]
                option_values = [option['value'] for option in options]
                value = cur_part if cur_part in option_values else self.agc_all_parts_value
                return options, value
            if not maker or not cartype:
                return [], None
            _, __, parts_all_by_mc = _maps_from_yaml()
            _, ____, parts_now_by_mc = _availability(start_date, end_date)
            parts = parts_now_by_mc.get((maker, cartype), []) or parts_all_by_mc.get((maker, cartype), [])
            opts = [{'label': p, 'value': p} for p in parts]
            val = cur_part if cur_part in parts else (parts[0] if parts else None)
            return opts, val
        

        # ---------- image ----------
        @self.app.callback(
            Output("part-image", "src"),
            Output("part-image", "style"),
            [Input("maker-dropdown", "value"),
             Input("cartype-dropdown", "value"),
             Input("part-dropdown", "value"),
             Input("view-selection", "value")]
        )
        def update_image(_maker, _cartype, selected_part, view):
            default_style = {'width': '100%', 'height': 'auto', 'margin-bottom': '20px'}
            if _maker == self.agc_maker or view == self.agc_view:
                return "", {'display': 'none'}
            if selected_part == self.agc_all_parts_value:
                return "", {'display': 'none'}
            image_path = f"assets/parts_img/{selected_part}.png" if selected_part else "assets/parts_img/not_found.png"
            if not os.path.exists(image_path):
                image_path = "assets/parts_img/not_found.png"
            return f"/{image_path}", default_style

        @self.app.callback(
            Output("summary-content", "children"),
            [
                Input('refresh-token', 'data'),
                Input("maker-dropdown", "value"),
                Input("cartype-dropdown", "value"),
                Input("part-dropdown", "value"),
                Input("date-picker", "start_date"),
                Input("date-picker", "end_date"),
                Input("view-selection", "value"),
            ]
        )
        def update_summary(_refresh_token, maker, cartype, selected_part, start_date, end_date, view):
            self.refresh_data()

            view_labels = {
                'inspection': '検査結果一覧',
                'ok_ng': 'OK/NG 品数',
                'kensa_time': '検査時間',
                'pitch_average': 'ピッチデータの詳細',
                self.agc_view: 'AGC ロット分析'
            }

            if view == self.agc_view:
                if maker != self.agc_maker:
                    return self._build_empty_state(
                        "AGC ロット分析は AGC 専用です",
                        "この表示を使う場合は、メーカーで AGC を選択してください。",
                        [
                            "メーカー選択で AGC を選びます。",
                            "表示タイプで AGC ロット分析 を選びます。",
                            "必要に応じて部品番号と期間を絞り込みます。"
                        ]
                    )

                date_context = self._get_date_context(maker)
                if not start_date:
                    start_date = date_context.get('default_start_date')
                if not end_date:
                    end_date = date_context.get('default_end_date')

                if not start_date or not end_date:
                    return self._build_empty_state(
                        "AGC データ範囲を取得できません",
                        "aikensa_agc.inspection_results の期間情報を読み込めませんでした。"
                    )

                lot_summary, _, daily_finished_df, _, daily_hourly_df = self._get_agc_lot_data(selected_part, start_date, end_date)
                selected_part_label = self._get_agc_part_label(selected_part)

                selection_row = html.Div([
                    self._build_filter_badge('メーカー', maker),
                    self._build_filter_badge('車種', cartype or self.agc_cartype),
                    self._build_filter_badge('製品', selected_part_label),
                    self._build_filter_badge('表示', view_labels.get(view, view or '未選択'), color='dark')
                ], style={'margin-bottom': '14px'})

                if lot_summary.empty:
                    return html.Div([
                        selection_row,
                        self._build_empty_state(
                            "AGC ロットデータがありません",
                            "現在の期間と部品条件ではロット集計の対象データが見つかりませんでした。",
                            [
                                "期間を広げて再確認してください。",
                                "製品を 全て に戻して確認してください。",
                                "temp から aikensa_agc.inspection_results への取込状態を確認してください。"
                            ]
                        )
                    ])

                total_lots = len(lot_summary)
                total_parts = int(lot_summary['total_parts'].sum())
                avg_sec_per_part = lot_summary['sec_per_part'].dropna().mean()
                avg_duration = lot_summary['duration_min'].mean()
                latest_row = lot_summary.sort_values('last_time').iloc[-1]
                total_daily_finished = int(daily_finished_df['finished_parts'].sum()) if not daily_finished_df.empty else 0
                avg_parts_per_hour = daily_hourly_df['parts_per_hour'].mean() if not daily_hourly_df.empty else None
                avg_theoretical_parts_per_hour = daily_hourly_df['theoretical_parts_per_hour'].mean() if not daily_hourly_df.empty else None

                kpi_row = dbc.Row([
                    dbc.Col(self._build_kpi_card('対象ロット数', f"{total_lots:,}", '#1d4ed8', subtitle='現在のフィルター条件で集計'), md=6, lg=3, className='mb-3'),
                    dbc.Col(self._build_kpi_card('総部品数', f"{total_parts:,}", '#15803d', subtitle=f"日次集計 {total_daily_finished:,}"), md=6, lg=3, className='mb-3'),
                    dbc.Col(self._build_kpi_card('平均秒/部品', f"{avg_sec_per_part:.2f}" if pd.notna(avg_sec_per_part) else '該当なし', '#7c3aed', subtitle=f"平均ロット時間 {avg_duration:.1f} 分"), md=6, lg=3, className='mb-3'),
                    dbc.Col(self._build_kpi_card('平均実績出来高/時間', f"{avg_parts_per_hour:.1f}" if pd.notna(avg_parts_per_hour) else '該当なし', '#0f766e', subtitle=f"理論 {avg_theoretical_parts_per_hour:.1f} / BREAK 5分換算" if pd.notna(avg_theoretical_parts_per_hour) else '3連続 5OK/0NG 基準 / BREAK 5分換算'), md=6, lg=3, className='mb-3'),
                ])

                period_label = f"{pd.to_datetime(start_date).strftime('%Y-%m-%d')} 〜 {pd.to_datetime(end_date).strftime('%Y-%m-%d')}"
                info_strip = dbc.Alert([
                    html.Span('表示期間: ', style={'font-weight': '700'}),
                    html.Span(period_label),
                    html.Span(' / データソース: ', style={'font-weight': '700', 'margin-left': '18px'}),
                    html.Span('MySQL aikensa_agc.inspection_results'),
                ], color='light', style={'border': '1px solid #d1d5db', 'margin-bottom': '0'})

                return html.Div([selection_row, kpi_row, info_strip])

            if self.combined_df is None or self.combined_df.empty:
                return self._build_empty_state(
                    "データがまだ読み込まれていません",
                    "MySQL またはローカルキャッシュから検査データを取得できていません。",
                    [
                        "接続設定と資格情報を確認してください。",
                        "キャッシュファイルが作成されているか確認してください。",
                        "最新データが投入されているか確認してください。"
                    ]
                )

            if not start_date:
                start_date = self.default_start_date
            if not end_date:
                end_date = self.default_end_date

            filtered_df = self.combined_df.copy()
            if selected_part and start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                filtered_df = filtered_df[
                    (filtered_df['partName'] == selected_part) &
                    (filtered_df['full_timestamp'] >= start_dt) &
                    (filtered_df['full_timestamp'] <= end_dt)
                ]
            else:
                start_dt = pd.to_datetime(start_date) if start_date else None
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59) if end_date else None
                if start_dt is not None and end_dt is not None:
                    filtered_df = filtered_df[
                        (filtered_df['full_timestamp'] >= start_dt) &
                        (filtered_df['full_timestamp'] <= end_dt)
                    ]

            status_series = filtered_df['status'].fillna('UNKNOWN').astype(str).str.upper() if 'status' in filtered_df.columns else pd.Series(dtype='object')
            total_records = len(filtered_df)
            ok_count = int((status_series == 'OK').sum()) if not status_series.empty else 0
            ng_count = int((status_series == 'NG').sum()) if not status_series.empty else 0
            abnormal_count = int((status_series != 'OK').sum()) if not status_series.empty else 0
            ok_rate = f"{(ok_count / total_records * 100):.1f}%" if total_records else "0.0%"
            latest_record = filtered_df['full_timestamp'].max() if total_records and 'full_timestamp' in filtered_df.columns else None
            freshness_minutes = None
            if self.last_update != datetime.min:
                freshness_minutes = max(0, int((datetime.now() - self.last_update).total_seconds() // 60))
            freshness_label = "更新直後" if freshness_minutes == 0 else f"{freshness_minutes} 分前更新" if freshness_minutes is not None else "更新情報なし"
            freshness_color = '#15803d' if freshness_minutes is not None and freshness_minutes <= 5 else '#b45309'

            selection_row = html.Div([
                self._build_filter_badge('メーカー', maker),
                self._build_filter_badge('車種', cartype),
                self._build_filter_badge('製品', selected_part),
                self._build_filter_badge('表示', view_labels.get(view, view or '未選択'), color='dark')
            ], style={'margin-bottom': '14px'})

            kpi_row = dbc.Row([
                dbc.Col(self._build_kpi_card('対象記録件数', f"{total_records:,}", '#1d4ed8', subtitle='現在のフィルター条件で集計'), md=6, lg=3, className='mb-3'),
                dbc.Col(self._build_kpi_card('OK率', ok_rate, '#15803d', subtitle=f"OK {ok_count:,} 件 / NG {ng_count:,} 件"), md=6, lg=3, className='mb-3'),
                dbc.Col(self._build_kpi_card('異常件数', f"{abnormal_count:,}", '#b91c1c', subtitle='NG / NOPART / MANUAL を含む'), md=6, lg=3, className='mb-3'),
                dbc.Col(self._build_kpi_card('最新検査時刻', latest_record.strftime('%Y-%m-%d %H:%M:%S') if latest_record is not None else 'データなし', freshness_color, subtitle=freshness_label), md=6, lg=3, className='mb-3'),
            ])

            period_label = f"{pd.to_datetime(start_date).strftime('%Y-%m-%d')} 〜 {pd.to_datetime(end_date).strftime('%Y-%m-%d')}" if start_date and end_date else '期間未指定'
            info_strip = dbc.Alert([
                html.Span('表示期間: ', style={'font-weight': '700'}),
                html.Span(period_label),
                html.Span(' / データ更新: ', style={'font-weight': '700', 'margin-left': '18px'}),
                html.Span(freshness_label),
            ], color='light', style={'border': '1px solid #d1d5db', 'margin-bottom': '0'})

            return html.Div([selection_row, kpi_row, info_strip])
        

        # ---------- Main dynamic content (inclusive end date) ----------
        @self.app.callback(
            Output("dynamic-content", "children"),
            [
                Input('refresh-token', 'data'),
                Input("maker-dropdown", "value"),      
                Input("cartype-dropdown", "value"),     
                Input("part-dropdown", "value"),
                Input("date-picker", "start_date"),
                Input("date-picker", "end_date"),
                Input("view-selection", "value"),
            ]
        )

        def update_content(_refresh_token, maker, cartype, selected_part, start_date, end_date, view):
            # you can ignore maker/cartype, but they must be in the signature
            if view == self.agc_view:
                if maker != self.agc_maker:
                    return self._build_empty_state(
                        "AGC ロット分析は AGC 専用です",
                        "メーカーで AGC を選択すると、ノートブック相当のロット可視化を表示します。",
                        [
                            "メーカーで AGC を選択してください。",
                            "必要に応じて部品番号を絞り込んでください。",
                            "期間を調整してロット傾向を確認してください。"
                        ]
                    )

                date_context = self._get_date_context(maker)
                if not start_date:
                    start_date = date_context.get('default_start_date')
                if not end_date:
                    end_date = date_context.get('default_end_date')

                if not start_date or not end_date:
                    return self._build_empty_state(
                        "AGC データ範囲を取得できません",
                        "aikensa_agc.inspection_results から期間情報を取得できませんでした。"
                    )

                lot_summary, display_summary, daily_finished_df, _daily_finished_pivot, daily_hourly_df = self._get_agc_lot_data(selected_part, start_date, end_date)
                return build_agc_lot_content(
                    lot_summary=lot_summary,
                    display_summary=display_summary,
                    daily_finished_df=daily_finished_df,
                    daily_hourly_df=daily_hourly_df,
                    selected_part_label=self._get_agc_part_label(selected_part),
                    build_kpi_card=self._build_kpi_card,
                    empty_state_builder=self._build_empty_state,
                )

            if maker == self.agc_maker:
                return self._build_empty_state(
                    "AGC では専用ビューを使用します",
                    "AGC メーカーのデータは MySQL の通常検査ビューではなく、AGC ロット分析で表示します。",
                    [
                        "表示タイプで AGC ロット分析 を選択してください。"
                    ]
                )

            if not selected_part or not start_date or not end_date:
                return self._build_empty_state(
                    "検索条件を選択してください",
                    "メーカー、車種、製品、日付を決めると、検査結果の一覧と分析が表示されます。",
                    [
                        "まずメーカーを選び、利用可能な車種を絞り込みます。",
                        "次に製品を選び、期間を指定します。",
                        "一覧、OK/NG、検査時間、ピッチ詳細を切り替えて確認できます。"
                    ]
                )

            self.refresh_data()
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date) + pd.to_timedelta(23, 'h') + pd.to_timedelta(59, 'm') + pd.to_timedelta(59, 's')

            filtered_df = self.combined_df[
                (self.combined_df['partName'] == selected_part) &
                (self.combined_df['full_timestamp'] >= start_date_dt) &
                (self.combined_df['full_timestamp'] <= end_date_dt)
            ]
            if filtered_df.empty:
                return self._build_empty_state(
                    "該当データがありません",
                    f"製品 {selected_part} の指定期間データが見つかりませんでした。",
                    [
                        "期間を広げて再確認してください。",
                        "同じメーカー・車種の別製品でデータがあるか確認してください。",
                        "設備停止やデータ取り込み遅延がないか確認してください。"
                    ]
                )


            # Handle the different view options (inspection, ok_ng, kensa_time, pitch_average)
            if view == 'inspection':
                columns_to_display = ['partName', "status", "NGreason", 'full_timestamp',
                                    'numofPart', 'currentnumofPart', 'kensainName',
                                    'cleaned_pitch', 'kensaTime', "PPMS"]
                # keep only existing columns (older rows safety)
                columns_to_display = [c for c in columns_to_display if c in filtered_df.columns]
                filtered_df = filtered_df[columns_to_display].sort_values(by='full_timestamp', ascending=False)
                filtered_df, pitch_columns = self._expand_pitch_result_columns(filtered_df, selected_part)
                custom_columns = [
                    {'name': '製品名', 'id': 'partName'},
                    {'name': '検査結果', 'id': 'status'},
                    {'name': 'NG理由', 'id': 'NGreason'},
                    {'name': '検査実施時間', 'id': 'full_timestamp'},
                    {'name': '本日数検査品数', 'id': 'numofPart'},
                    {'name': '現時検査品数', 'id': 'currentnumofPart'},
                    {'name': '検査員番号', 'id': 'kensainName'},
                    {'name': 'サイクルタイム', 'id': 'kensaTime'},
                    {'name': 'PPMS番号', 'id': 'PPMS'}
                ]
                insert_at = 7
                for index, pitch_column in enumerate(pitch_columns):
                    custom_columns.insert(insert_at + index, {'name': pitch_column, 'id': pitch_column})
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

                abnormal_df = filtered_df.copy()
                if 'status' in abnormal_df.columns:
                    abnormal_df['status'] = abnormal_df['status'].fillna('').astype(str).str.upper()
                    abnormal_df = abnormal_df[abnormal_df['status'].isin(['NG', 'NOPART', 'MANUAL'])]

                abnormal_rate = f"{(len(abnormal_df) / len(filtered_df) * 100):.1f}%" if len(filtered_df) else "0.0%"
                top_reason_label = translated_reasons[0] if translated_reasons else '異常理由なし'
                top_reason_count = int(counts[0]) if counts else 0
                latest_abnormal = abnormal_df['full_timestamp'].max() if not abnormal_df.empty and 'full_timestamp' in abnormal_df.columns else None

                abnormal_summary = dbc.Row([
                    dbc.Col(self._build_kpi_card('異常率', abnormal_rate, '#b91c1c', subtitle=f"異常 {len(abnormal_df):,} 件 / 全体 {len(filtered_df):,} 件"), md=4, className='mb-3'),
                    dbc.Col(self._build_kpi_card('最多異常理由', top_reason_label, '#c2410c', subtitle=f"{top_reason_count:,} 件"), md=4, className='mb-3'),
                    dbc.Col(self._build_kpi_card('直近異常時刻', latest_abnormal.strftime('%Y-%m-%d %H:%M:%S') if latest_abnormal is not None else '異常なし', '#7c3aed', subtitle='NG / NOPART / MANUAL'), md=4, className='mb-3'),
                ])

                abnormal_table = self._build_empty_state(
                    '異常履歴はありません',
                    '現在の条件では NG / NOPART / MANUAL の履歴は見つかりませんでした。'
                )
                if not abnormal_df.empty:
                    abnormal_df = abnormal_df.sort_values('full_timestamp', ascending=False).copy()
                    if 'NGreason' in abnormal_df.columns:
                        abnormal_df['NGreason_ja'] = abnormal_df['NGreason'].fillna('').astype(str).apply(lambda value: translation.get(value, value))
                    else:
                        abnormal_df['NGreason_ja'] = ''

                    abnormal_columns = [
                        'full_timestamp', 'status', 'NGreason_ja', 'kensainName', 'PPMS', 'partName'
                    ]
                    abnormal_columns = [col for col in abnormal_columns if col in abnormal_df.columns]
                    column_defs = [
                        {'name': '検査時刻', 'id': 'full_timestamp'},
                        {'name': '状態', 'id': 'status'},
                        {'name': '異常理由', 'id': 'NGreason_ja'},
                        {'name': '検査員番号', 'id': 'kensainName'},
                        {'name': 'PPMS番号', 'id': 'PPMS'},
                        {'name': '製品名', 'id': 'partName'}
                    ]
                    column_defs = [col for col in column_defs if col['id'] in abnormal_columns]
                    abnormal_table = dash_table.DataTable(
                        data=abnormal_df[abnormal_columns].head(25).to_dict('records'),
                        columns=column_defs,
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_header={'textAlign': 'center', 'fontWeight': '700'},
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{status} = "NG"'}, 'backgroundColor': '#fee2e2', 'color': '#991b1b'},
                            {'if': {'filter_query': '{status} = "NOPART"'}, 'backgroundColor': '#fef3c7', 'color': '#92400e'},
                            {'if': {'filter_query': '{status} = "MANUAL"'}, 'backgroundColor': '#ede9fe', 'color': '#5b21b6'}
                        ]
                    )

                return html.Div([
                    abnormal_summary,
                    dcc.Graph(figure=stacked_fig),
                    dcc.Graph(figure=pareto_fig),
                    dcc.Graph(figure=resultpitch_fig),
                    html.H4('最近の異常履歴', style={'margin-top': '24px', 'margin-bottom': '12px'}),
                    abnormal_table
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

                if actual_pitch_count == 0:
                    return self._build_empty_state(
                        "ピッチ定義がありません",
                        f"製品 {selected_part} には有効なピッチ設定がありません。",
                        [
                            "parts.yaml の pitch_count を確認してください。",
                            "parts.yaml の基準ピッチと公差設定に漏れがないか確認してください。"
                        ]
                    )

                pitch_graphs = []
                total_measurements = 0
                total_tolerance_breaches = 0
                total_three_sigma_breaches = 0
                worst_nominal_gap = 0.0
                worst_pitch_label = '-'

                for i in range(actual_pitch_count):
                    pitch_column = f'Pitch {i+1}'
                    pitch_label = f'ピッチ {i+1}'
                    temp_df = expanded_pitch_df[[pitch_column, 'Date']].copy()
                    temp_df = temp_df[temp_df[pitch_column] != 0]

                    if temp_df.empty:
                        pitch_graphs.append(
                            self._build_empty_state(
                                f'{pitch_label} の測定値がありません',
                                '現在の条件では有効なピッチ測定値を取得できませんでした。'
                            )
                        )
                        continue

                    series = pd.to_numeric(temp_df[pitch_column], errors='coerce').dropna()
                    if series.empty:
                        pitch_graphs.append(
                            self._build_empty_state(
                                f'{pitch_label} の数値化に失敗しました',
                                '測定値フォーマットを確認してください。'
                            )
                        )
                        continue

                    fig = px.box(
                        temp_df,
                        x='Date',
                        y=pitch_column,
                        title=f'{pitch_label} の日別分布',
                        labels={'Date': '日付', pitch_column: 'ピッチ(mm)'}
                    )
                    nominal = nominal_pitch[i] if i < len(nominal_pitch) else None
                    tol = tolerance[i] if i < len(tolerance) else None

                    pitch_mean = float(series.mean())
                    pitch_std = float(series.std(ddof=0)) if len(series) > 1 else 0.0
                    sigma_upper = pitch_mean + (3 * pitch_std)
                    sigma_lower = pitch_mean - (3 * pitch_std)

                    tolerance_breach_count = 0
                    max_nominal_gap = 0.0
                    if nominal is not None:
                        nominal_gap = (series - nominal).abs()
                        max_nominal_gap = float(nominal_gap.max()) if not nominal_gap.empty else 0.0
                        worst_nominal_gap = max(worst_nominal_gap, max_nominal_gap)
                        if max_nominal_gap == worst_nominal_gap:
                            worst_pitch_label = pitch_label
                    if nominal is not None and tol is not None:
                        lower_tol = nominal - tol
                        upper_tol = nominal + tol
                        tolerance_breach_count = int(((series < lower_tol) | (series > upper_tol)).sum())
                        fig.add_shape(
                            type="rect",
                            xref="paper",
                            x0=0,
                            x1=1,
                            yref="y",
                            y0=lower_tol,
                            y1=upper_tol,
                            fillcolor="rgba(34, 197, 94, 0.10)",
                            line=dict(width=0),
                            layer="below"
                        )
                    else:
                        lower_tol = None
                        upper_tol = None

                    three_sigma_breach_count = int(((series < sigma_lower) | (series > sigma_upper)).sum()) if pitch_std > 0 else 0
                    total_measurements += int(len(series))
                    total_tolerance_breaches += tolerance_breach_count
                    total_three_sigma_breaches += three_sigma_breach_count

                    if nominal is not None:
                        fig.add_hline(y=nominal, line_color="#2563eb", line_dash="dash", annotation_text="基準値", annotation_position="top left")
                    if lower_tol is not None and upper_tol is not None:
                        fig.add_hline(y=upper_tol, line_color="#16a34a", line_dash="dot", annotation_text="公差上限", annotation_position="top left")
                        fig.add_hline(y=lower_tol, line_color="#16a34a", line_dash="dot", annotation_text="公差下限", annotation_position="bottom left")

                    fig.add_hline(y=pitch_mean, line_color="#111827", line_dash="solid", annotation_text="平均値", annotation_position="top right")
                    if pitch_std > 0:
                        fig.add_hline(y=sigma_upper, line_color="#ea580c", line_dash="dash", annotation_text="+3σ", annotation_position="top right")
                        fig.add_hline(y=sigma_lower, line_color="#ea580c", line_dash="dash", annotation_text="-3σ", annotation_position="bottom right")

                    daily_means = temp_df.groupby('Date')[pitch_column].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=daily_means['Date'],
                        y=daily_means[pitch_column],
                        mode='markers',
                        marker=dict(symbol='x', size=9, color='black'),
                        name='日別平均'
                    ))
                    fig.update_layout(
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        margin=dict(l=50, r=30, t=80, b=40),
                        title=(
                            f"{pitch_label} の日別分布"
                            f"<br><sup>平均 {pitch_mean:.3f} mm / σ {pitch_std:.3f} / 3σ管理範囲 [{sigma_lower:.3f}, {sigma_upper:.3f}]</sup>"
                        )
                    )

                    metric_cards = dbc.Row([
                        dbc.Col(self._build_kpi_card('平均', f'{pitch_mean:.3f} mm', '#111827', subtitle=f'σ {pitch_std:.3f}'), md=3, className='mb-2'),
                        dbc.Col(self._build_kpi_card('3σ管理範囲', f'{sigma_lower:.3f} ~ {sigma_upper:.3f}', '#ea580c', subtitle='平均値 ± 3σ'), md=3, className='mb-2'),
                        dbc.Col(self._build_kpi_card('公差外件数', f'{tolerance_breach_count:,}', '#16a34a', subtitle='基準値 ± 公差'), md=3, className='mb-2'),
                        dbc.Col(self._build_kpi_card('3σ外件数', f'{three_sigma_breach_count:,}', '#dc2626', subtitle=f'最大偏差 {max_nominal_gap:.3f} mm'), md=3, className='mb-2'),
                    ])

                    pitch_graphs.append(
                        dbc.Card(
                            dbc.CardBody([
                                metric_cards,
                                dcc.Graph(figure=fig)
                            ]),
                            style={'margin-bottom': '20px', 'border-radius': '14px', 'box-shadow': '0 8px 24px rgba(15, 23, 42, 0.08)'}
                        )
                    )

                overall_cards = dbc.Row([
                    dbc.Col(self._build_kpi_card('総測定点数', f'{total_measurements:,}', '#1d4ed8', subtitle='現在のピッチ表示範囲'), md=3, className='mb-3'),
                    dbc.Col(self._build_kpi_card('公差外総数', f'{total_tolerance_breaches:,}', '#16a34a', subtitle='基準値 ± 公差'), md=3, className='mb-3'),
                    dbc.Col(self._build_kpi_card('3σ外総数', f'{total_three_sigma_breaches:,}', '#dc2626', subtitle='工程ばらつき基準'), md=3, className='mb-3'),
                    dbc.Col(self._build_kpi_card('最大基準値偏差', f'{worst_nominal_gap:.3f} mm', '#7c3aed', subtitle=worst_pitch_label), md=3, className='mb-3'),
                ])

                explanation = dbc.Alert([
                    html.Span('青線: 基準値 / 緑帯: 公差範囲 / 橙線: 平均値 ± 3σ / 黒マーカー: 日別平均', style={'font-weight': '600'})
                ], color='light', style={'border': '1px solid #d1d5db'})

                return html.Div([overall_cards, explanation] + pitch_graphs)

            return "Invalid view selected."

        # ---------- Excel export (uses inclusive end date) ----------
        @self.app.callback(
            Output("download-dataframe-xlsx", "data"),
            Input("download-excel", "n_clicks"),
            State("part-dropdown", "value"),
            State("date-picker", "start_date"),
            State("date-picker", "end_date"),
            State("view-selection", "value"),
            State("maker-dropdown", "value"),
            prevent_initial_call=True
        )
        def download_excel(n_clicks, selected_part, start_date, end_date, view, maker):
            if not n_clicks:
                return no_update

            if view == self.agc_view:
                if maker != self.agc_maker or not start_date or not end_date:
                    return no_update

                df_to_export = self._build_agc_lot_export_df(selected_part, start_date, end_date)
                if df_to_export.empty:
                    return no_update

                def to_excel(bytes_io):
                    with pd.ExcelWriter(bytes_io, engine="xlsxwriter") as writer:
                        df_to_export.to_excel(writer, index=False, sheet_name="AGC Lot Ichiran")

                return dcc.send_bytes(to_excel, "agc_lot_ichiran.xlsx")

            if view != "inspection" or not selected_part or not start_date or not end_date:
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
            df_to_export, pitch_columns = self._expand_pitch_result_columns(df_to_export, selected_part)
            ordered_columns = ['partName', 'numofPart', 'currentnumofPart', 'kensainName']
            ordered_columns.extend(pitch_columns)
            ordered_columns.extend(['full_timestamp', 'kensaTime', 'PPMS'])
            cols_present = [c for c in ordered_columns if c in df_to_export.columns]
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
