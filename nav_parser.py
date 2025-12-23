from __future__ import annotations

import re
from datetime import date
from typing import Optional, List, Dict, Tuple

import pandas as pd
from dateutil import parser as dateparser


ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
EMAIL_DATE_RE = re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")
NAV_DATE_FULL_RE = re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")
NAV_DATE_NOYEAR_RE = re.compile(r"\b(\d{1,2}\.\d{1,2}\.)\b")
PRICE_CCY_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2,8})?)\s*(USD|EUR|HUF)\b", re.IGNORECASE)


def _normalize_lines(text: str) -> List[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    for ln in lines:
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out


def _parse_number(s: str) -> float:
    s = str(s).strip()
    # "4,919.0000" -> 4919.0000
    if "," in s and "." in s:
        return float(s.replace(",", ""))
    # "31,7100" -> 31.7100
    if "," in s and "." not in s:
        return float(s.replace(",", "."))
    return float(s)


def extract_email_date(pasted_text: str) -> Optional[date]:
    m = EMAIL_DATE_RE.search(pasted_text)
    if not m:
        return None
    try:
        dt = dateparser.parse(m.group(1), dayfirst=True)
        return dt.date()
    except Exception:
        return None


def infer_nav_date_from_ddmm(nav_day: int, nav_month: int, email_dt: date) -> date:
    # Your rule: if email in January and NAV month is December => previous year.
    nav_year = email_dt.year - 1 if (email_dt.month == 1 and nav_month == 12) else email_dt.year
    return date(nav_year, nav_month, nav_day)


def parse_nav_date(window: List[str], email_dt: date) -> Optional[date]:
    joined = " ".join(window)

    m_full = NAV_DATE_FULL_RE.search(joined)
    if m_full:
        try:
            dt = dateparser.parse(m_full.group(1), dayfirst=True)
            return dt.date()
        except Exception:
            pass

    m_ny = NAV_DATE_NOYEAR_RE.search(joined)
    if m_ny:
        ddmm = m_ny.group(1)  # "30.12."
        try:
            tmp = dateparser.parse(ddmm + "2000", dayfirst=True).date()
            return infer_nav_date_from_ddmm(tmp.day, tmp.month, email_dt)
        except Exception:
            return None

    return None


def find_price_ccy(window: List[str]) -> Optional[Tuple[float, str]]:
    joined = " ".join(window)
    m = PRICE_CCY_RE.search(joined)
    if not m:
        return None
    nav_str, ccy = m.group(1), m.group(2).upper()
    return _parse_number(nav_str), ccy


def guess_fund_name(lines: List[str], isin_idx: int) -> Optional[str]:
    bad = {"name", "isin", "type", "exchange", "date", "time", "last", "curr", "chg", "chart", "fonds", "funds", "settings"}
    for j in range(isin_idx - 1, max(-1, isin_idx - 6), -1):
        ln = lines[j]
        low = ln.lower()
        if any(tok in low for tok in bad):
            continue
        if ISIN_RE.search(ln):
            continue
        if len(ln) >= 6:
            return ln
    return None


def parse_baha_paste(pasted_text: str, only_isins: Optional[set[str]] = None) -> Tuple[pd.DataFrame, date]:
    email_dt = extract_email_date(pasted_text)
    if email_dt is None:
        raise ValueError("No full email date (dd.mm.yyyy) found in pasted text. Paste the email including header.")

    lines = _normalize_lines(pasted_text)
    records: List[Dict[str, object]] = []

    for idx, ln in enumerate(lines):
        isins = ISIN_RE.findall(ln)
        if not isins:
            continue

        for isin in isins:
            isin = isin.upper()
            if only_isins and isin not in only_isins:
                continue

            fund_name = guess_fund_name(lines, idx)
            window = lines[idx: min(len(lines), idx + 15)]

            nav_dt = parse_nav_date(window, email_dt=email_dt)
            nav_ccy = find_price_ccy(window)
            excerpt = " | ".join(window[:10])

            if nav_dt and nav_ccy:
                nav, ccy = nav_ccy
                records.append(
                    {
                        "nav_date": nav_dt,
                        "isin": isin,
                        "fund_name": fund_name,
                        "nav": float(nav),
                        "currency": ccy,
                        "raw_excerpt": excerpt,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        return df, email_dt

    df = df.sort_values(["nav_date", "isin"]).drop_duplicates(["nav_date", "isin"], keep="last")
    return df, email_dt
