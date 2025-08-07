import pandas as pd
import io
from typing import List, Dict, Any
import matplotlib.pyplot as plt

def create_excel_dashboard(
    filename: str,
    dataframes: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    charts: List[Dict[str, Any]]
) -> None:
    """
    Creates an Excel dashboard with DataFrames, metrics, and charts positioned as specified.
    
    Parameters
    ----------
    filename : str
        Path of the Excel file to create.
    dataframes : list of dict
        Each dict requires:
            - 'df': pandas.DataFrame
            - 'sheet': str, sheet name
            - 'startrow': int
            - 'startcol': int
    metrics : list of dict
        Each dict requires:
            - 'name': str, label for the metric
            - 'value': Any, the metric value
            - 'sheet': str, sheet name
            - 'row': int
            - 'col': int
    charts : list of dict
        Each dict requires:
            - 'fig': matplotlib.figure.Figure
            - 'sheet': str, sheet name
            - 'cell': str, Excel cell reference (e.g. 'E2')
    """
    # Use pandas ExcelWriter with xlsxwriter engine
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook  = writer.book
        # Write all DataFrames
        for item in dataframes:
            df = item['df']
            sheet = item['sheet']
            startrow = item.get('startrow', 0)
            startcol = item.get('startcol', 0)
            df.to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=True)
        
        # Write metrics
        for m in metrics:
            sheet = m['sheet']
            worksheet = writer.sheets.setdefault(sheet, workbook.add_worksheet(sheet))
            label = m['name']
            val = m['value']
            row, col = m['row'], m['col']
            worksheet.write(row, col, label)
            worksheet.write(row, col+1, val)
        
        # Insert charts
        for ch in charts:
            sheet = ch['sheet']
            worksheet = writer.sheets.setdefault(sheet, workbook.add_worksheet(sheet))
            cell = ch['cell']
            fig = ch['fig']
            # Save figure to a buffer.
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            worksheet.insert_image(cell, '', {'image_data': buf, 'x_scale': 1.0, 'y_scale': 1.0})
            plt.close(fig)

# Example Usage
if __name__ == "__main__":
    # Sample data
    df1 = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=5, freq="D"),
        "Value": [100, 110, 120, 130, 140]
    })
    df1.set_index("Date", inplace=True)
    
    # Sample chart
    fig, ax = plt.subplots()
    df1.plot(ax=ax)
    ax.set_title("Sample Time Series")
    
    config_dfs = [
        {"df": df1, "sheet": "Summary", "startrow": 1, "startcol": 1}
    ]
    config_metrics = [
        {"name": "Mean Value", "value": df1["Value"].mean(), "sheet": "Summary", "row": 0, "col": 1}
    ]
    config_charts = [
        {"fig": fig, "sheet": "Charts", "cell": "B2"}
    ]
    
    create_excel_dashboard("dashboard.xlsx", config_dfs, config_metrics, config_charts)
    print("Excel dashboard created: dashboard.xlsx")
