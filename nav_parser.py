import re
from datetime import date
from typing import Optional, List, Tuple

import pandas as pd
from dateutil import parser as dateparser

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
DATE_DDMM_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.\b")
PRICE_LINE_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2,8})?)\s*(USD|EUR|HUF)\b", re.IGNORECASE)

SENT_KEYS = [
    "sent",            # EN
    "odoslané",        # SK
    "odoslane",        # SK no diacritics
    "elküldve",        # HU
    "elkuldve",        # HU no diacritics
]

HEADER_STOP_KEYS = [
    "to", "komu", "címzett", "cimzett",
    "subject", "predmet", "tárgy", "targy",
    "from", "od", "feladó", "felado",
]

def _strip_accents(s: str) -> str:
    try:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s

def _clean_markdown(text: str) -> str:
    # normalize newlines and NBSP
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")

    # remove bold markers
    text = text.replace("**", "")

    # convert markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # remove markdown image lines like: [image](...) or ![...](...)
    text = re.sub(r"^\s*!\[.*?\]\(.*?\)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\[image\]\(.*?\)\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text

def _normalize_lines(text: str) -> list[str]:
    text = _clean_markdown(text)
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        # drop markdown headings (#, ###)
        ln = re.sub(r"^\s*#{1,6}\s*", "", ln)

        # normalize whitespace (incl. tabs)
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out

def extract_email_date(pasted_text: str) -> Optional[date]:
    """
    Robust: finds Sent:/Odoslané:/Elküldve: even if inline on same line.
    Accepts English month names (e.g., "Tuesday, December 23, 2025 9:02 AM").
    """
    text = _clean_markdown(pasted_text)

    # Work on an accent-stripped copy for matching, but keep original slice for parsing
    text_norm = _strip_accents(text).lower()

    # Build a regex to capture "Sent: <date text>" until next header (To/Subject/From) or end
    sent_keys_re = "|".join([re.escape(_strip_accents(k).lower()) for k in SENT_KEYS])
    stop_keys_re = "|".join([re.escape(_strip_accents(k).lower()) for k in HEADER_STOP_KEYS])

    pattern = re.compile(
        rf"(?:^|[\n\s])(?:{sent_keys_re})\s*:\s*(.+?)(?=(?:[\n\s]+(?:{stop_keys_re})\s*:)|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    m = pattern.search(text_norm)
    if m:
        # Find the same span in the original (non-normalized) text by using the matched group content length
        # We’ll parse the matched substring from the original text by re-searching "Sent:" in original.
        # Simple approach: extract from original with a second regex (case-insensitive) without accent stripping.
        pattern_orig = re.compile(
            rf"(?:^|[\n\s])(?:{'|'.join(SENT_KEYS)})\s*:\s*(.+?)(?=(?:[\n\s]+(?:{'|'.join(HEADER_STOP_KEYS)})\s*:)|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m2 = pattern_orig.search(text)
        raw = (m2.group(1) if m2 else m.group(1)).strip()

        try:
            dt = dateparser.parse(raw, dayfirst=True, fuzzy=True)
            if dt:
                return dt.date()
        except Exception:
            pass

    # Fallback 1: look for dd.mm.yyyy or dd/mm/yyyy
    m3 = re.search(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b", text)
    if m3:
        try:
            dt = dateparser.parse(m3.group(1), dayfirst=True, fuzzy=True)
            if dt:
                return dt.date()
        except Exception:
            pass

    # Fallback 2: any substring with a 4-digit year near it
    m4 = re.search(r"(.{0,40}\b\d{4}\b.{0,40})", text)
    if m4:
        try:
            dt = dateparser.parse(m4.group(1), fuzzy=True)
            if dt:
                return dt.date()
        except Exception:
            pass

    return None

def _parse_number(s: str) -> float:
    s = str(s).strip()
    if "," in s and "." in s:
        # 4,919.0000 -> 4919.0000
        return float(s.replace(",", ""))
    if "," in s and "." not in s:
        # 31,7100 -> 31.7100
        return float(s.replace(",", "."))
    return float(s)

def infer_nav_date_from_ddmm(nav_day: int, nav_month: int, email_dt: date) -> date:
    nav_year = email_dt.year - 1 if (email_dt.month == 1 and nav_month == 12) else email_dt.year
    return date(nav_year, nav_month, nav_day)

def guess_fund_name(lines: List[str], isin_idx: int) -> Optional[str]:
    bad = {
        "name", "isin", "type", "exchange", "date", "time",
        "last", "curr", "chg", "chart", "fonds", "funds", "settings",
        "securities price notification"
    }
    for j in range(isin_idx - 1, max(-1, isin_idx - 8), -1):
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
        raise ValueError(
            "Email date not found. Paste the header including Sent:/Odoslané:/Elküldve: (with year) "
            "or include a full dd.mm.yyyy date."
        )

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

        fund_name = guess_fund_name(lines, i)

        window = lines[i: min(len(lines), i + 14)]

        # 1) Find NAV date line index (dd.mm.)
        nav_dt = None
        date_idx = None
        for idx_w, w in enumerate(window):
            dm = DATE_DDMM_RE.search(w)
            if dm:
                d = int(dm.group(1))
                m = int(dm.group(2))
                nav_dt = infer_nav_date_from_ddmm(d, m, email_dt)
                date_idx = idx_w
                break

        # 2) Find NAV value+ccy AFTER the date line (avoids picking header values)
        nav_val = None
        nav_ccy = None
        search_from = (date_idx + 1) if date_idx is not None else 0

        for w in window[search_from:]:
            pm = PRICE_LINE_RE.search(w)
            if pm:
                nav_val = _parse_number(pm.group(1))
                nav_ccy = pm.group(2).upper()
                break

        if nav_dt and nav_val is not None and nav_ccy:
            excerpt = " | ".join(window[:8])
            records.append({
                "nav_date": nav_dt,
                "isin": isin,
                "fund_name": fund_name,
                "nav": float(nav_val),
                "currency": nav_ccy,
                "raw_excerpt": excerpt,
            })

        i += 1

    df = pd.DataFrame(records)
    if df.empty:
        return df, email_dt

    df = df.sort_values(["nav_date", "isin"]).drop_duplicates(["nav_date", "isin"], keep="last")
    return df, email_dt
