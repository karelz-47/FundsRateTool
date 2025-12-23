from __future__ import annotations

import io
import re
from datetime import date
from typing import Dict, Optional

import pandas as pd


def _to_float(s: str) -> float:
    s = str(s).replace("\xa0", "").strip()
    if not s:
        raise ValueError("Empty numeric field.")
    return float(s.replace(",", "."))


def parse_tb_rates_from_csv_bytes(file_bytes: bytes) -> Dict[str, float]:
    """
    Expected TB CSV export (semicolon-separated) with columns:
      Kód;Devíza nákup;Devíza predaj;Devíza stred;...
    We extract only HUF and USD rows.
    """
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";", encoding="utf-8-sig")

    required = {"Kód", "Devíza nákup", "Devíza predaj", "Devíza stred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in TB CSV: {sorted(missing)}. Found: {list(df.columns)}")

    df["Kód"] = df["Kód"].astype(str).str.strip().str.upper()

    def row(code: str):
        r = df.loc[df["Kód"] == code]
        if r.empty:
            raise ValueError(f"Currency '{code}' not found in TB CSV.")
        return r.iloc[0]

    huf = row("HUF")
    usd = row("USD")

    return {
        "huf_buy": _to_float(huf["Devíza nákup"]),
        "huf_sell": _to_float(huf["Devíza predaj"]),
        "huf_mid": _to_float(huf["Devíza stred"]),
        "usd_buy": _to_float(usd["Devíza nákup"]),
        "usd_sell": _to_float(usd["Devíza predaj"]),
        "usd_mid": _to_float(usd["Devíza stred"]),
    }


def try_parse_date_from_filename(filename: str) -> Optional[date]:
    """
    Optional helper if you later name files with a date.
    Supports:
      - YYYY-MM-DD
      - DD.MM.YYYY
    """
    name = filename.strip()

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", name)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", name)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    return None
