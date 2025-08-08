# pip install xlsxwriter matplotlib pandas numpy  # if needed

from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Tuple, Union
import io
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ============================== #
#   EXCEL DASHBOARD EXPORTER     #
# ============================== #

def export_excel_dashboard(
    filepath: str,
    dataframes: Dict[str, pd.DataFrame] | None = None,
    metrics: Dict[str, Union[str, float, int]] | Iterable[Tuple[str, Union[str, float, int]]] | None = None,
    figures: Optional[List[Figure]] = None,
    charts_per_row: int = 2,
    dfs_per_row: int = 1,
    table_style: str = "Table Style Medium 9",
    image_scale: float = 0.9,
    autofit_sample: int = 500,
    data_sheet_name: str = "Data & Metrics",
    charts_sheet_name: str = "Charts",
    date_format: str = "yyyy-mm-dd",
    percent_keywords: Tuple[str, ...] = ("pct", "perc", "%", "prob", "rate", "yield"),
) -> None:
    """
    Create a neat Excel file with:
      - One sheet for metrics + DataFrames, laid out automatically with spacing, titles, tables, and auto-fit columns.
      - One sheet for charts (matplotlib figures), tiled in a grid with spacing and scaling.

    Parameters
    ----------
    filepath : str
        Output .xlsx path.
    dataframes : dict[name -> DataFrame], optional
    metrics : dict[name -> value] or iterable of (name, value), optional
    figures : list of matplotlib.figure.Figure, optional
    charts_per_row : int
        How many charts per row on the Charts sheet.
    dfs_per_row : int
        How many DataFrames per row on the Data & Metrics sheet (grid layout). 1 = stacked vertically.
    table_style : str
        Excel table style name.
    image_scale : float
        Scale applied to inserted chart images.
    autofit_sample : int
        Max rows sampled for column auto-fit (speed/robustness).
    data_sheet_name : str
    charts_sheet_name : str
    date_format : str
    percent_keywords : tuple of substrings that hint a column is a percent (format 0.00%).

    Returns
    -------
    None (writes the Excel file).
    """
    dataframes = dataframes or {}
    # Normalize metrics into list of (key, value) tuples for stable order:
    if metrics is None:
        metric_items = []
    elif isinstance(metrics, dict):
        metric_items = list(metrics.items())
    else:
        metric_items = list(metrics)

    # If figures=None, grab currently open figs:
    if figures is None:
        figures = [plt.figure(num) for num in plt.get_fignums()]

    # ---- Writer / workbook / sheets
    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Styles
        fmt_title = workbook.add_format({"bold": True, "font_size": 12, "valign": "vcenter"})
        fmt_section = workbook.add_format({"bold": True, "font_size": 14, "valign": "vcenter"})
        fmt_wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
        fmt_header = workbook.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        fmt_general = workbook.add_format({"border": 1})
        fmt_int = workbook.add_format({"num_format": "0", "border": 1})
        fmt_float = workbook.add_format({"num_format": "0.00", "border": 1})
        fmt_pct = workbook.add_format({"num_format": "0.00%", "border": 1})
        fmt_date = workbook.add_format({"num_format": date_format, "border": 1})

        # -------------- Data & Metrics sheet --------------
        ws_data = workbook.add_worksheet(_sanitize_sheetname(data_sheet_name))
        ws_data.set_zoom(110)

        cur_row = 0
        cur_col = 0

        # Metrics block (if any)
        if metric_items:
            ws_data.write(cur_row, cur_col, "Metrics", fmt_section)
            cur_row += 2

            # Put metrics into a small DataFrame for uniform table rendering
            mdf = pd.DataFrame(metric_items, columns=["Metric", "Value"])
            _write_table(
                worksheet=ws_data,
                top_row=cur_row,
                left_col=cur_col,
                df=mdf,
                workbook=workbook,
                table_style=table_style,
                header_format=fmt_header,
                cell_formats=_column_formats_for_df(
                    mdf, fmt_int, fmt_float, fmt_pct, fmt_date, fmt_general, percent_keywords
                ),
            )
            # Auto-fit 2 columns of metrics:
            _autofit_columns(ws_data, cur_row, cur_col, mdf, max_width=60, sample=autofit_sample)
            cur_row += len(mdf) + 4  # add spacing after metrics

        # DataFrames grid
        if dataframes:
            ws_data.write(cur_row, cur_col, "DataFrames", fmt_section)
            cur_row += 2

            # Pre-compute each DF's rendered height/width for grid packing
            names = list(dataframes.keys())
            dfs = [dataframes[k] for k in names]
            heights = [len(df) + 3 for df in dfs]  # title + header + data
            widths = [max(df.shape[1], 1) + 2 for df in dfs]  # columns + padding

            # Track the row offset per grid row using the max height in that row
            grid_row = 0
            idx = 0
            # Maintain per-grid-row current top row
            top_row_for_grid_row = cur_row

            while idx < len(dfs):
                # Compute block for this grid row: items idx..idx+dfs_per_row-1
                block = list(zip(names[idx: idx + dfs_per_row], dfs[idx: idx + dfs_per_row],
                                 heights[idx: idx + dfs_per_row], widths[idx: idx + dfs_per_row]))
                # Place each DF in the row from left to right
                left_col = 0
                max_height = 0

                for (name, df, h, w) in block:
                    # Section title
                    ws_data.write(top_row_for_grid_row, left_col, str(name), fmt_title)
                    # Actual table underneath
                    table_top = top_row_for_grid_row + 1
                    _write_table(
                        worksheet=ws_data,
                        top_row=table_top,
                        left_col=left_col,
                        df=df,
                        workbook=workbook,
                        table_style=table_style,
                        header_format=fmt_header,
                        cell_formats=_column_formats_for_df(
                            df, fmt_int, fmt_float, fmt_pct, fmt_date, fmt_general, percent_keywords
                        ),
                    )
                    _autofit_columns(ws_data, table_top, left_col, df, max_width=60, sample=autofit_sample)

                    left_col += w
                    max_height = max(max_height, h)

                # Move to next grid row
                top_row_for_grid_row += max_height + 2  # spacing between DF rows
                grid_row += 1
                idx += dfs_per_row

            # Freeze top-left corner below the "DataFrames" title if we wrote DFS
            ws_data.freeze_panes(cur_row, 0)

        # -------------- Charts sheet --------------
        ws_charts = workbook.add_worksheet(_sanitize_sheetname(charts_sheet_name))
        ws_charts.set_zoom(110)

        if figures:
            # Title
            ws_charts.write(0, 0, "Charts", fmt_section)
            # Start placing images a couple rows below
            chart_row_cursor = 2
            chart_col_cursor = 0

            # Determine grid steps based on image pixel size -> row/column translation
            # (xlsxwriter row-height default ~20 px, col-width ~64 px at width=8.43; we’ll compute per-image)
            for i, fig in enumerate(figures):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                buf.seek(0)

                # Rough dimensions in pixels (for spacing)
                try:
                    from PIL import Image  # not strictly required; only for spacing fidelity
                    img = Image.open(buf)
                    px_w, px_h = img.size
                    buf.seek(0)
                except Exception:
                    # Fallback
                    px_w, px_h = 1000, 600

                # Compute row/col increments to avoid overlap after scaling
                step_rows = int(math.ceil((px_h * image_scale) / 20.0)) + 4
                step_cols = int(math.ceil((px_w * image_scale) / 64.0)) + 1

                r = chart_row_cursor
                c = chart_col_cursor

                ws_charts.insert_image(
                    r, c,
                    f"chart_{i}.png",
                    {"image_data": buf, "x_scale": image_scale, "y_scale": image_scale}
                )

                # Advance grid
                if ((i + 1) % charts_per_row) != 0:
                    chart_col_cursor += step_cols
                else:
                    chart_col_cursor = 0
                    chart_row_cursor += step_rows

        # Done; writer saves on exit


# ----------------- Helpers ----------------- #

def _sanitize_sheetname(name: str) -> str:
    bad = set(r"[]:*?/\\")
    cleaned = "".join(ch for ch in name if ch not in bad)
    if not cleaned:
        cleaned = "Sheet1"
    return cleaned[:31]

def _write_table(
    worksheet,
    top_row: int,
    left_col: int,
    df: pd.DataFrame,
    workbook,
    table_style: str,
    header_format,
    cell_formats: List
):
    """
    Renders a pandas DataFrame as a styled Excel table at (top_row, left_col).
    """
    nrows, ncols = df.shape
    # Convert to values (handles NaN nicely)
    data = df.values.tolist()
    columns = [{"header": str(col), "header_format": header_format, "format": cell_formats[j]} for j, col in enumerate(df.columns)]

    # Add the table
    worksheet.add_table(
        top_row,
        left_col,
        top_row + nrows,
        left_col + ncols - 1,
        {
            "data": data,
            "columns": columns,
            "style": table_style,
            "autofilter": True,
            "banded_columns": False,
        },
    )

def _column_formats_for_df(
    df: pd.DataFrame,
    fmt_int, fmt_float, fmt_pct, fmt_date, fmt_general, percent_keywords: Tuple[str, ...]
) -> List:
    """
    Choose a sensible format per column.
    """
    formats = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            formats.append(fmt_date)
        elif pd.api.types.is_integer_dtype(s):
            formats.append(fmt_int)
        elif pd.api.types.is_float_dtype(s):
            # Heuristic: treat as percent if mostly between 0 and 1 and/or name hints percent
            name_l = str(col).lower()
            looks_percent = any(k in name_l for k in percent_keywords)
            if not looks_percent:
                # sample to avoid scanning huge columns
                sample = s.dropna()
                if len(sample) > 0:
                    sample = sample.sample(min(1000, len(sample)), random_state=1)
                    frac_01 = np.mean((sample >= 0.0) & (sample <= 1.0))
                    looks_percent = frac_01 > 0.85
            formats.append(fmt_pct if looks_percent else fmt_float)
        else:
            formats.append(fmt_general)
    return formats

def _autofit_columns(worksheet, top_row: int, left_col: int, df: pd.DataFrame, max_width: int = 60, sample: int = 500):
    """
    Auto-fit columns for a block of df placed at (top_row, left_col).
    Width heuristic = max(len(str(cell)) of header + sampled cells) + padding, then clipped to max_width.
    """
    padding = 2
    nrows, ncols = df.shape
    # Headers
    widths = [len(str(c)) + padding for c in df.columns]

    # Sample rows for speed
    if len(df) > sample:
        sample_rows = df.iloc[np.linspace(0, len(df) - 1, sample, dtype=int)]
    else:
        sample_rows = df

    for j, col in enumerate(df.columns):
        col_vals = sample_rows[col].astype("string").fillna("")
        max_len = col_vals.map(len).max() if len(col_vals) else 0
        widths[j] = min(max(widths[j], max_len + padding), max_width)

    for j, w in enumerate(widths):
        worksheet.set_column(left_col + j, left_col + j, float(w))




# Assemble your objects
dfs = {
    "Attribution Results": df_attr,     # pd.DataFrame
    "Factor Z-Scores": df_z,            # pd.DataFrame
    "Signal PnL (monthly)": df_pnl      # pd.DataFrame
}

metrics = {
    "Run Date": pd.Timestamp.today().normalize(),
    "Universe": "UST 2s/5s/10s/30s",
    "Backtest Start": pd.Timestamp("2013-01-01"),
    "Backtest End": pd.Timestamp("2025-08-01"),
    "Hit Rate (TP>0)": 0.6123,
    "IR (monthly)": 1.27,
}

# Collect charts (any matplotlib Figure objects)
figs = [fig1, fig2, fig3]  # or leave None to auto-grab currently open figs

export_excel_dashboard(
    filepath="TermPremium_Dashboard.xlsx",
    dataframes=dfs,
    metrics=metrics,
    figures=figs,
    charts_per_row=2,   # tile charts 2 per row
    dfs_per_row=1,      # stack dataframes vertically; set 2 for a 2-up grid
    image_scale=0.9
)

