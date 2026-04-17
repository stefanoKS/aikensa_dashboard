from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import plotly.graph_objects as go


def _apply_dashboard_layout(fig: go.Figure, title: str, yaxis_title: str, xaxis_title: str = "ロット番号") -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=70, b=50),
    )
    fig.update_xaxes(type="category", tickangle=-30, showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.25)", zerolinecolor="rgba(148, 163, 184, 0.25)")
    return fig


def _section_title(title: str, subtitle: str):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "1.15rem", "fontWeight": "800", "color": "#111827"}),
            html.Div(subtitle, style={"fontSize": "0.92rem", "color": "#4b5563", "marginTop": "4px"}),
        ],
        style={"marginBottom": "14px"},
    )


def _content_card(children, accent: str = "#8B0000", extra_style: dict | None = None):
    style = {
        "borderRadius": "18px",
        "boxShadow": "0 12px 30px rgba(15, 23, 42, 0.08)",
        "marginBottom": "20px",
        "border": "1px solid rgba(148, 163, 184, 0.22)",
        "borderTop": f"4px solid {accent}",
        "background": "linear-gradient(180deg, #ffffff 0%, #fcfcfd 100%)",
    }
    if extra_style:
        style.update(extra_style)
    return dbc.Card(dbc.CardBody(children), style=style)


def _build_stacked_ok_ng_chart(plot_df: pd.DataFrame) -> go.Figure:
    x_values = plot_df["lotDisplay"] if "lotDisplay" in plot_df.columns else plot_df["lotNumber"]
    totals = plot_df["ok_total"] + plot_df["ng_total"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=plot_df["ok_total"],
            name="OK合計",
            marker_color="#15803d",
            text=plot_df["ok_total"],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=plot_df["ng_total"],
            name="NG合計",
            marker_color="#b91c1c",
            text=[f"{value:.0f}" for value in totals],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="stack")
    return _apply_dashboard_layout(fig, "ロット別 OK/NG 合計", "数量")


def _build_single_bar_chart(
    plot_df: pd.DataFrame,
    value_column: str,
    title: str,
    yaxis_title: str,
    color: str,
    text_format: str,
) -> go.Figure:
    x_values = plot_df["lotDisplay"] if "lotDisplay" in plot_df.columns else plot_df["lotNumber"]
    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=plot_df[value_column],
            marker_color=color,
            text=[text_format.format(value) for value in plot_df[value_column]],
            textposition="outside",
            name=yaxis_title,
        )
    )
    return _apply_dashboard_layout(fig, title, yaxis_title)


def _build_sec_per_part_chart(plot_df: pd.DataFrame) -> go.Figure:
    x_values = plot_df["lotDisplay"] if "lotDisplay" in plot_df.columns else plot_df["lotNumber"]
    sec_values = plot_df["sec_per_part"]
    labels = ["該当なし" if pd.isna(value) else f"{value:.2f}" for value in sec_values]
    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=sec_values.fillna(0),
            marker_color="#7c3aed",
            text=labels,
            textposition="outside",
            name="秒/部品",
        )
    )
    return _apply_dashboard_layout(fig, "部品1個あたり秒数", "秒/部品")


def _build_time_to_next_chart(plot_df: pd.DataFrame) -> go.Figure:
    x_values = plot_df["lotDisplay"] if "lotDisplay" in plot_df.columns else plot_df["lotNumber"]
    values = plot_df["time_to_next_lot_min"]
    labels = []
    hover_labels = []
    break_flags = plot_df.get("is_break_to_next_lot", pd.Series(False, index=plot_df.index))
    for value, is_break in zip(values, break_flags):
        if pd.isna(value):
            labels.append("該当なし")
            hover_labels.append("該当なし")
            continue
        if bool(is_break):
            labels.append(f"BREAK({value:.1f})")
            hover_labels.append(f"BREAK({value:.1f} 分)")
            continue
        labels.append(f"{value:.1f}")
        hover_labels.append(f"{value:.1f} 分")
    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=values.fillna(0),
            marker_color="#0f766e",
            text=labels,
            textposition="outside",
            name="分",
            customdata=hover_labels,
            hovertemplate="ロット %{x}<br>%{customdata}<extra></extra>",
        )
    )
    return _apply_dashboard_layout(fig, "次ロット切替までの時間", "分")


def _build_daily_ng_rate_chart(daily_hourly_df: pd.DataFrame) -> go.Figure:
    if daily_hourly_df.empty:
        return go.Figure()

    plot_df = daily_hourly_df.copy()
    plot_df["inspection_date"] = pd.to_datetime(plot_df["inspection_date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["inspection_date"]).sort_values("inspection_date")
    plot_df["date_label"] = plot_df["inspection_date"].dt.strftime("%Y-%m-%d")
    values = plot_df["ng_rate_pct"]
    labels = ["該当なし" if pd.isna(value) else f"{value:.2f}%" for value in values]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["date_label"],
            y=values,
            mode="lines+markers+text",
            line=dict(color="#b91c1c", width=3),
            marker=dict(size=8, color="#b91c1c"),
            text=labels,
            textposition="top center",
            name="NG率",
            hovertemplate="日付 %{x}<br>NG率 %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_xaxes(
        type="category",
        tickangle=-30,
        categoryorder="array",
        categoryarray=plot_df["date_label"].tolist(),
    )
    fig.update_yaxes(ticksuffix="%")
    return _apply_dashboard_layout(fig, "日別 NG率 推移", "NG率 (%)", xaxis_title="日付")


def _build_daily_finished_chart(daily_finished_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    sorted_daily_df = daily_finished_df.copy()
    sorted_daily_df["inspection_date"] = pd.to_datetime(sorted_daily_df["inspection_date"], errors="coerce")
    sorted_daily_df = sorted_daily_df.dropna(subset=["inspection_date"])
    sorted_daily_df = sorted_daily_df.sort_values(["inspection_date", "partLabel"])
    category_dates = sorted_daily_df["inspection_date"].dt.strftime("%Y-%m-%d").drop_duplicates().tolist()

    part_colors = {
        "J59JLH": "#1d4ed8",
        "J59JRH": "#15803d",
        "J30LH": "#c2410c",
        "J30RH": "#7c3aed",
    }
    for part_label in sorted_daily_df["partLabel"].dropna().unique().tolist():
        part_df = sorted_daily_df[sorted_daily_df["partLabel"] == part_label].copy()
        part_df["inspection_date_label"] = part_df["inspection_date"].dt.strftime("%Y-%m-%d")
        fig.add_trace(
            go.Bar(
                x=part_df["inspection_date_label"],
                y=part_df["finished_parts"],
                name=part_label,
                marker_color=part_colors.get(str(part_label), "#334155"),
                text=[f"{value:.0f}" for value in part_df["finished_parts"]],
                textposition="outside",
                hovertemplate="日付 %{x}<br>品番 %{fullData.name}<br>完了数 %{y}<extra></extra>",
            )
        )

    daily_totals = (
        sorted_daily_df.groupby("inspection_date", as_index=False)["finished_parts"]
        .sum()
        .sort_values("inspection_date")
    )
    fig.update_layout(
        annotations=[
            dict(
                x=row["inspection_date"].strftime("%Y-%m-%d"),
                y=1.04,
                xref="x",
                yref="paper",
                text=f"合計 {row['finished_parts']:.0f}",
                showarrow=False,
                font=dict(size=12, color="#111827"),
                align="center",
            )
            for _, row in daily_totals.iterrows()
        ]
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(
        type="category",
        tickangle=-30,
        categoryorder="array",
        categoryarray=category_dates,
    )
    return _apply_dashboard_layout(fig, "品番別 日次完了数", "完了数", xaxis_title="日付")


def _build_daily_heatmap(daily_finished_df: pd.DataFrame) -> go.Figure:
    if daily_finished_df.empty:
        return go.Figure()

    pivot_df = (
        daily_finished_df.pivot(index="partLabel", columns="inspection_date", values="finished_parts")
        .fillna(0)
        .sort_index()
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot_df.values,
            x=[str(value) for value in pivot_df.columns.tolist()],
            y=pivot_df.index.tolist(),
            colorscale=[
                [0.0, "#fff7ed"],
                [0.25, "#fdba74"],
                [0.5, "#fb923c"],
                [0.75, "#ea580c"],
                [1.0, "#9a3412"],
            ],
            colorbar=dict(title="完了数"),
            hovertemplate="%{y}<br>%{x}<br>完了数 %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="日次品番ミックス ヒートマップ",
        xaxis_title="日付",
        yaxis_title="品番",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=70, b=50),
    )
    return fig


def _build_daily_parts_per_hour_chart(daily_hourly_df: pd.DataFrame) -> go.Figure:
    if daily_hourly_df.empty:
        return go.Figure()

    plot_df = daily_hourly_df.copy()
    plot_df["inspection_date"] = pd.to_datetime(plot_df["inspection_date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["inspection_date"]).sort_values("inspection_date")
    plot_df["date_label"] = plot_df["inspection_date"].dt.strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["date_label"],
            y=plot_df["parts_per_hour"],
            marker_color="#0f766e",
            text=[f"{value:.1f}" for value in plot_df["parts_per_hour"]],
            textposition="outside",
            name="実績出来高/時間",
            hovertemplate="日付 %{x}<br>実績出来高/時間 %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date_label"],
            y=plot_df["parts_per_hour"],
            mode="lines+markers",
            line=dict(color="#134e4a", width=3),
            marker=dict(size=8, color="#134e4a"),
            name="実績トレンド",
            hovertemplate="日付 %{x}<br>実績出来高/時間 %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date_label"],
            y=plot_df["theoretical_parts_per_hour"],
            mode="lines+markers",
            line=dict(color="#991b1b", width=3, dash="dash"),
            marker=dict(size=9, color="#991b1b", symbol="diamond"),
            name="理論出来高/時間",
            hovertemplate="日付 %{x}<br>理論出来高/時間 %{y:.1f}<extra></extra>",
            visible="legendonly",
        )
    )
    fig.update_xaxes(
        type="category",
        tickangle=-30,
        categoryorder="array",
        categoryarray=plot_df["date_label"].tolist(),
    )
    return _apply_dashboard_layout(fig, "日別 実績/理論 出来高比較", "部品数/時間", xaxis_title="日付")


def _build_part_mix_summary_table(daily_finished_df: pd.DataFrame) -> dash_table.DataTable:
    summary_df = (
        daily_finished_df.groupby(["partName", "partLabel"], as_index=False)
        .agg(
            total_finished=("finished_parts", "sum"),
            average_per_day=("finished_parts", "mean"),
            peak_day_output=("finished_parts", "max"),
            total_ok=("ok_total", "sum"),
            total_ng=("ng_total", "sum"),
            total_lots=("lots", "sum"),
        )
        .sort_values("partName", key=lambda series: series.astype(int))
    )
    summary_df["average_per_day"] = summary_df["average_per_day"].round(1)

    return dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[
            {"name": "品番", "id": "partLabel"},
            {"name": "完了総数", "id": "total_finished"},
            {"name": "日平均", "id": "average_per_day"},
            {"name": "最大日次出来高", "id": "peak_day_output"},
            {"name": "OK", "id": "total_ok"},
            {"name": "NG", "id": "total_ng"},
            {"name": "ロット数", "id": "total_lots"},
        ],
        page_size=10,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={"textAlign": "center", "fontWeight": "700", "backgroundColor": "#f3f4f6"},
        style_cell={"textAlign": "center", "padding": "10px", "minWidth": "110px"},
    )


def _build_agc_header(selected_part_label: str, lot_summary: pd.DataFrame, daily_finished_df: pd.DataFrame):
    total_lots = len(lot_summary)
    total_finished = int(daily_finished_df["finished_parts"].sum()) if not daily_finished_df.empty else 0
    date_min = daily_finished_df["inspection_date"].min() if not daily_finished_df.empty else None
    date_max = daily_finished_df["inspection_date"].max() if not daily_finished_df.empty else None
    coverage = f"{date_min} 〜 {date_max}" if date_min is not None and date_max is not None else "日次データなし"

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("AGC ロット分析", style={"fontSize": "0.82rem", "fontWeight": "800", "letterSpacing": "0.14em", "color": "#fecaca"}),
                    html.H3(f"{selected_part_label} のロット推移と日次完了数", style={"fontWeight": "800", "marginTop": "8px", "marginBottom": "8px", "color": "#ffffff"}),
                ], lg=8),
                dbc.Col([
                    html.Div(f"{total_finished:,}", style={"fontSize": "2rem", "fontWeight": "800", "lineHeight": "1.0", "color": "#ffffff", "textAlign": "right"}),
                    html.Div("期間内の完了数", style={"fontSize": "0.85rem", "color": "#fecaca", "textAlign": "right"}),
                    html.Div(f"{total_lots:,} ロットを追跡", style={"fontSize": "0.92rem", "color": "rgba(255,255,255,0.92)", "marginTop": "10px", "textAlign": "right"}),
                    html.Div(coverage, style={"fontSize": "0.82rem", "color": "rgba(255,255,255,0.75)", "textAlign": "right"}),
                ], lg=4),
            ])
        ]),
        style={
            "marginBottom": "22px",
            "border": "none",
            "borderRadius": "22px",
            "background": "linear-gradient(135deg, #7f1d1d 0%, #991b1b 48%, #111827 100%)",
            "boxShadow": "0 20px 40px rgba(127, 29, 29, 0.25)",
        },
    )


def build_agc_lot_content(
    lot_summary: pd.DataFrame,
    display_summary: pd.DataFrame,
    daily_finished_df: pd.DataFrame,
    daily_hourly_df: pd.DataFrame,
    selected_part_label: str,
    build_kpi_card,
    empty_state_builder,
):
    if lot_summary.empty:
        return empty_state_builder(
            "AGC ロットデータがありません",
            "指定した期間と部品条件ではロット集計を作成できませんでした。",
            [
                "期間を広げて再確認してください。",
                "部品を 全て に戻して傾向を確認してください。",
                "aikensa_agc.inspection_results に最新データが取り込まれているか確認してください。",
            ],
        )

    plot_df = lot_summary.copy()
    total_ok = int(plot_df["ok_total"].sum())
    total_ng = int(plot_df["ng_total"].sum())
    total_parts = total_ok + total_ng
    total_ng_rate = (100 * total_ng / total_parts) if total_parts else None
    average_duration = plot_df["duration_min"].mean()
    average_sec_per_part = plot_df["sec_per_part"].dropna().mean()
    unique_parts = daily_finished_df["partName"].nunique() if not daily_finished_df.empty else 0
    best_day = None if daily_finished_df.empty else daily_finished_df.groupby("inspection_date")["finished_parts"].sum().idxmax()
    best_day_output = 0 if daily_finished_df.empty else int(daily_finished_df.groupby("inspection_date")["finished_parts"].sum().max())
    average_parts_per_hour = daily_hourly_df["parts_per_hour"].mean() if not daily_hourly_df.empty else None

    metric_row = dbc.Row([
        dbc.Col(build_kpi_card("OK 合計", f"{total_ok:,}", "#15803d", subtitle=f"NG {total_ng:,} / 部品 {selected_part_label}"), md=6, lg=3, className="mb-3"),
        dbc.Col(build_kpi_card("NG率", f"{total_ng_rate:.2f}%" if pd.notna(total_ng_rate) else "該当なし", "#b91c1c", subtitle="NG ÷ 総部品数"), md=6, lg=3, className="mb-3"),
        dbc.Col(build_kpi_card("平均ロット時間", f"{average_duration:.1f} 分", "#c2410c", subtitle="ロット先頭検査から最終検査まで"), md=6, lg=3, className="mb-3"),
        dbc.Col(build_kpi_card("平均秒/部品", f"{average_sec_per_part:.2f}" if pd.notna(average_sec_per_part) else "該当なし", "#7c3aed", subtitle="ロット時間 ÷ 総部品数"), md=6, lg=3, className="mb-3"),
    ])

    throughput_row = dbc.Row([
        dbc.Col(build_kpi_card("対象部品種類", f"{unique_parts:,}", "#1d4ed8", subtitle="ダミー品番は除外"), md=6, lg=6, className="mb-3"),
        dbc.Col(build_kpi_card("最大日次出来高", f"{best_day_output:,}", "#b45309", subtitle=str(best_day) if best_day is not None else "該当なし"), md=6, lg=6, className="mb-3"),
    ])

    hourly_row = dbc.Row([
        dbc.Col(build_kpi_card("平均実績出来高/時間", f"{average_parts_per_hour:.1f}" if pd.notna(average_parts_per_hour) else "該当なし", "#0f766e", subtitle="次ロットまで 30 分超は BREAK として 5 分換算"), md=12, lg=12, className="mb-3"),
    ])

    description = _content_card([
        _section_title("AGC 生産サマリー", "ノートブック相当のロット分析に加えて、AGC 各品番の日次完了数も確認できます。"),
        html.Div([
            dbc.Badge("ロット別 OK/NG", color="danger", className="me-2"),
            dbc.Badge("日別 NG率", color="danger", className="me-2"),
            dbc.Badge("総部品数", color="primary", className="me-2"),
            dbc.Badge("ロット時間", color="warning", className="me-2"),
            dbc.Badge("BREAK補正", color="success", className="me-2"),
            dbc.Badge("秒/部品", color="secondary", className="me-2"),
            dbc.Badge("品番別日次出来高", color="dark", className="me-2"),
            dbc.Badge("理論出来高/時間", color="danger", className="me-2"),
        ]),
    ], accent="#991b1b")

    stacked_fig = _build_stacked_ok_ng_chart(plot_df)
    duration_fig = _build_single_bar_chart(
        plot_df,
        value_column="duration_min",
        title="ロット実作業時間",
        yaxis_title="分",
        color="#c2410c",
        text_format="{:.1f}",
    )
    sec_per_part_fig = _build_sec_per_part_chart(plot_df)
    time_to_next_fig = _build_time_to_next_chart(plot_df)
    daily_finished_fig = _build_daily_finished_chart(daily_finished_df)
    daily_parts_per_hour_fig = _build_daily_parts_per_hour_chart(daily_hourly_df)
    daily_ng_rate_fig = _build_daily_ng_rate_chart(daily_hourly_df)

    table_columns = [
        {"name": "品番", "id": "partLabel"},
        {"name": "ロット番号", "id": "lotNumber"},
        {"name": "OK", "id": "ok_total"},
        {"name": "NG", "id": "ng_total"},
        {"name": "総部品数", "id": "total_parts"},
        {"name": "時間当たり(本/時間)", "id": "parts_per_hour"},
        {"name": "秒/部品", "id": "sec_per_part"},
        {"name": "開始時刻", "id": "first_time"},
        {"name": "終了時刻", "id": "last_time"},
        {"name": "実作業時間(分)", "id": "duration_min"},
        {"name": "次ロットまで(分)", "id": "time_to_next_lot_min"},
    ]

    summary_table = dash_table.DataTable(
        data=display_summary.to_dict("records"),
        columns=table_columns,
        page_size=15,
        sort_action="native",
        style_table={"overflowX": "auto", "width": "100%"},
        style_header={"textAlign": "center", "fontWeight": "700", "backgroundColor": "#f3f4f6", "fontSize": "11px", "padding": "6px"},
        style_header_conditional=[
            {"if": {"column_id": "parts_per_hour"}, "backgroundColor": "#dbeafe", "color": "#1e3a8a"},
        ],
        style_cell={"textAlign": "center", "padding": "5px 6px", "minWidth": "72px", "width": "72px", "maxWidth": "120px", "fontSize": "11px", "lineHeight": "1.15", "whiteSpace": "normal"},
        style_cell_conditional=[
            {"if": {"column_id": "partLabel"}, "minWidth": "68px", "width": "68px", "maxWidth": "84px"},
            {"if": {"column_id": "lotNumber"}, "minWidth": "118px", "width": "118px", "maxWidth": "132px"},
            {"if": {"column_id": "first_time"}, "minWidth": "128px", "width": "128px", "maxWidth": "140px"},
            {"if": {"column_id": "last_time"}, "minWidth": "128px", "width": "128px", "maxWidth": "140px"},
            {"if": {"column_id": "parts_per_hour"}, "minWidth": "88px", "width": "88px", "maxWidth": "96px"},
            {"if": {"column_id": "time_to_next_lot_min"}, "minWidth": "88px", "width": "88px", "maxWidth": "96px"},
            {"if": {"column_id": "parts_per_hour"}, "backgroundColor": "#eff6ff", "color": "#1d4ed8", "fontWeight": "700"},
        ],
    )

    export_button = html.Button(
        "Excelエクスポート",
        id="download-excel",
        n_clicks=0,
        className="btn btn-primary",
        style={"marginBottom": "16px"},
    )

    return html.Div([
        _build_agc_header(selected_part_label, lot_summary, daily_finished_df),
        metric_row,
        throughput_row,
        hourly_row,
        description,
        _content_card([
            _section_title("ロット推移", "ロット単位の OK/NG、出来高、時間、処理速度を確認します。"),
            dcc.Graph(figure=stacked_fig),
        ], accent="#991b1b"),
        _content_card([dcc.Graph(figure=duration_fig)], accent="#c2410c"),
        dbc.Row([
            dbc.Col(_content_card([dcc.Graph(figure=sec_per_part_fig)], accent="#7c3aed"), lg=6),
            dbc.Col(_content_card([dcc.Graph(figure=time_to_next_fig)], accent="#0f766e"), lg=6),
        ]),
        _content_card([
            _section_title("日次完了数", "AGC 各品番ごとに、日別の完了部品数を確認します。"),
            dcc.Graph(figure=daily_finished_fig),
            html.Div("実績出来高/時間は当日の連続検査間隔を積み上げて計算し、次ロットまで 30 分を超える空き時間は BREAK とみなして 5 分換算します。理論出来高/時間は 3 連続の 5OK / 0NG が成立した区間から各ロット最短サイクルを取り、その日の日平均で算出します。ロット実作業時間は同一ロット内で 10 分を超える停止時間を除外しています。", style={"fontSize": "0.9rem", "color": "#4b5563", "marginBottom": "10px"}),
            dcc.Graph(figure=daily_ng_rate_fig),
            dcc.Graph(figure=daily_parts_per_hour_fig),
            html.H5("品番別出来高サマリー", style={"marginTop": "8px", "marginBottom": "12px", "fontWeight": "700"}),
            _build_part_mix_summary_table(daily_finished_df),
        ], accent="#b45309"),
        _content_card([
            _section_title("AGC ロット一覧", "ダッシュボード上で詳細ロット集計を確認できます。"),
                export_button,
                html.H4("AGC ロット集計表", style={"margin-bottom": "16px", "font-weight": "700"}),
                summary_table,
        ], accent="#334155"),
    ])