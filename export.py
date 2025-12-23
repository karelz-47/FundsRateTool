from __future__ import annotations

import io
import pandas as pd
from openpyxl import Workbook

from config import HEADER_ROW1, HEADER_ROW2, HEADER_ROW3, SERIES_ORDER


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out.insert(0, "Date", out.index.date)
    return out.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    ws.append(HEADER_ROW1)
    ws.append(HEADER_ROW2)
    ws.append(HEADER_ROW3)

    for idx, row in df.iterrows():
        ws.append([idx.to_pydatetime()] + [float(row[c]) for c in SERIES_ORDER])

    # Basic column widths
    for col in ws.columns:
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = 18

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()
