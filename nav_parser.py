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
SENT_KEYS = [
    "sent",            # EN
    "odoslané",        # SK
    "odoslane",        # SK without diacritics
    "elküldve",        # HU
    "elkuldve",        # HU without diacritics
]
DATE_DDMM_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.\b")
PRICE_LINE_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2,8})?)\s*(USD|EUR|HUF)\b", re.IGNORECASE)
FUNDS_WORDS = {"funds", "fonds"}  # tolerate EN/FR/DE-ish variants; we use this as weak hint



def _strip_accents(s: str) -> str:
    try:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s
        

def _normalize_lines(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    for ln in lines:
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out


def _parse_number(s: str) -> float:
    s = str(s).strip()
    if "," in s and "." in s:
        # 4,919.0000 -> 4919.0000
        return float(s.replace(",", ""))
    if "," in s and "." not in s:
        # 31,7100 -> 31.7100
        return float(s.replace(",", "."))
    return float(s)


def extract_email_date(pasted_text: str):
    text = pasted_text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Try explicit "Sent:" line (EN/SK/HU)
    for line in text.split("\n"):
        low = _strip_accents(line).lower().strip()
        # match "Sent:" / "Odoslané:" / "Elküldve:"
        if any(low.startswith(k + ":") for k in SENT_KEYS):
            raw = line.split(":", 1)[1].strip()
            # robust parsing: fuzzy handles extra words/timezone
            try:
                dt = dateparser.parse(raw, dayfirst=True, fuzzy=True)
                if dt:
                    return dt.date()
            except Exception:
                pass

    # 2) Fallback: any dd.mm.yyyy in header area
    m = re.search(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b", text)
    if m:
        dt = dateparser.parse(m.group(1), dayfirst=True, fuzzy=True)
        if dt:
            return dt.date()

    return None
    
def infer_nav_date_from_ddmm(nav_day: int, nav_month: int, email_dt: date) -> date:
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


def parse_baha_paste(pasted_text: str, only_isins: Optional[set[str]] = None):
    email_dt = extract_email_date(pasted_text)
    if email_dt is None:
        raise ValueError("Email date not found. Paste the header including 'Sent:' / 'Odoslané:' / 'Elküldve:' or a full dd.mm.yyyy date.")

    lines = _normalize_lines(pasted_text)
    records = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        m_isin = ISIN_RE.search(ln)
        if not m_isin:
            i += 1
            continue

        isin = m_isin.group(0).upper()
        if only_isins and isin not in only_isins:
            i += 1
            continue

        # Fund name is typically the previous non-empty line
        fund_name = None
        if i - 1 >= 0:
            fund_name = lines[i - 1]

        # Expect structure:
        # [name] - [currency]  (currency at end is helpful but not mandatory)
        # [ISIN] / Funds
        # Fonds 22.12.
        # 37.4900 USD -0.4200
        # -1.11%

        nav_dt = None
        nav_val = None
        nav_ccy = None

        # Search forward a limited window (block)
        window = lines[i: min(len(lines), i + 10)]

        # 1) Find NAV date line containing dd.mm.
        for w in window:
            dm = DATE_DDMM_RE.search(w)
            if dm:
                d = int(dm.group(1))
                m = int(dm.group(2))
                nav_dt = infer_nav_date_from_ddmm(d, m, email_dt)
                break

        # 2) Find NAV value + currency line (first occurrence after the date line ideally)
        # We'll scan the window and pick the first plausible NAV.
        for w in window:
            pm = PRICE_LINE_RE.search(w)
            if pm:
                nav_val = _parse_number(pm.group(1))
                nav_ccy = pm.group(2).upper()
                break

        if nav_dt and nav_val is not None and nav_ccy:
            excerpt = " | ".join(window[:6])
            records.append({
                "nav_date": nav_dt,
                "isin": isin,
                "fund_name": fund_name,
                "nav": float(nav_val),
                "currency": nav_ccy,
                "raw_excerpt": excerpt
            })

        i += 1

    df = pd.DataFrame(records)
    if df.empty:
        return df, email_dt

    df = df.sort_values(["nav_date", "isin"]).drop_duplicates(["nav_date", "isin"], keep="last")
    return df, email_dt
