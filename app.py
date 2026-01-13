from __future__ import annotations

import base64
import hashlib
import json
import re
import csv
import io
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import backfill_legacy
from db_upsert import upsert_published_rates
from sqlalchemy import (
    select,
    Column,
    Integer,
    String,
    Date,
    DateTime,
    Float,          # or use Numeric instead (recommended for money-like series)
    Numeric,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from db import (
    SessionLocal,
    init_db,
    FxRate,
    FundNav,
    CalcRun,
    CalcDaily,
    compute_input_hash,
    PublishedRate,
)
from fx_import import parse_tb_rates_from_csv_bytes, try_parse_date_from_filename
from nav_parser import parse_baha_paste
from calc import compute_outputs
from export import to_csv_bytes, to_xlsx_bytes
from config import NAV_CURRENCY, SERIES_ORDER, TR_YEARLY_YIELD_DEFAULT, FUND_LABELS, ROUND_DECIMALS, APOLLON_FUND_CODE_BY_ISIN


# =============================================================================
# Paths / Assets
# =============================================================================

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_BIG = ASSETS_DIR / "FundRatesTool_logo.png"
LOGO_SMALL = ASSETS_DIR / "FundRatesTool_logo_small.png"


# =============================================================================
# i18n (loads EN.json, with flexible filename support incl. "EN (1).json")
# =============================================================================

def _find_locale_file(lang: str) -> Optional[Path]:
    """
    Looks for locale files in:
      - ./locales/<LANG>.json
      - ./<LANG>.json
    plus tolerant glob matches:
      - ./locales/<LANG>*.json (e.g., "EN (1).json")
      - ./<LANG>*.json
    """
    candidates = [
        BASE_DIR / "locales" / f"{lang}.json",
        BASE_DIR / f"{lang}.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Tolerate filenames like "EN (1).json"
    glob_candidates = []
    glob_candidates.extend(sorted((BASE_DIR / "locales").glob(f"{lang}*.json")))
    glob_candidates.extend(sorted(BASE_DIR.glob(f"{lang}*.json")))
    for p in glob_candidates:
        if p.exists():
            return p
    return None


def _load_translations(lang: str) -> Dict[str, str]:
    p = _find_locale_file(lang)
    if p is None:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def t(key: str, default: Optional[str] = None) -> str:
    """
    Translation helper with safe fallback:
      - if key in loaded translations -> return it
      - else -> default (if provided)
      - else -> a readable fallback based on key
    """
    tr: Dict[str, str] = st.session_state.get("_tr", {})
    if key in tr and isinstance(tr[key], str):
        return tr[key]
    if default is not None:
        return default
    # Human-ish fallback
    return key.replace("_", " ").strip()


def _init_i18n():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "EN"
    lang = st.session_state["lang"]
    st.session_state["_tr"] = _load_translations(lang)


# =============================================================================
# Page config (avoid crashing if icon is missing/bad)
# =============================================================================

def _is_png_file(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        b = path.read_bytes()
        return b.startswith(b"\x89PNG\r\n\x1a\n")
    except Exception:
        return False


page_icon_value: Any = None
if _is_png_file(LOGO_SMALL):
    page_icon_value = str(LOGO_SMALL)

st.set_page_config(
    page_title="Fund Rates Tool",
    page_icon=page_icon_value,  # safe
    layout="wide",
)

_init_i18n()


# =============================================================================
# Header
# =============================================================================

def render_header_with_logo(title_text: str):
    # Use base64 HTML embed to avoid PIL decoding issues
    if not LOGO_BIG.exists() or not _is_png_file(LOGO_BIG):
        st.title(title_text)
        return

    b64 = base64.b64encode(LOGO_BIG.read_bytes()).decode("utf-8")
    html = f"""
    <div style="
        display:flex;
        align-items:center;
        gap:1.25rem;
        margin: 0 0 1.25rem 0;
    ">
      <img src="data:image/png;base64,{b64}"
           style="height: 90px; width:auto; display:block;" />
      <h1 style="margin:0; font-size: 2.2rem; font-weight: 700;">
        {title_text}
      </h1>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# DB init
# =============================================================================

@st.cache_resource
def _init():
    init_db()
    return True


_init()


# =============================================================================
# Loaders
# =============================================================================

def _load_fx(session) -> pd.DataFrame:
    rows = session.execute(select(FxRate)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["rate_date", "huf_buy", "huf_sell", "huf_mid", "usd_buy", "usd_sell", "usd_mid"])
    return (
        pd.DataFrame(
            [
                {
                    "rate_date": r.rate_date,
                    "huf_buy": r.huf_buy,
                    "huf_sell": r.huf_sell,
                    "huf_mid": r.huf_mid,
                    "usd_buy": r.usd_buy,
                    "usd_sell": r.usd_sell,
                    "usd_mid": r.usd_mid,
                    "source": r.source,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
        )
        .sort_values("rate_date")
    )


def _load_nav(session) -> pd.DataFrame:
    rows = session.execute(select(FundNav)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["nav_date", "isin", "nav", "currency", "fund_name", "source_email_date", "raw_hash", "created_at"])

    return (
        pd.DataFrame(
            [
                {
                    "nav_date": r.nav_date,
                    "isin": r.isin,
                    "nav": r.nav,
                    "currency": r.currency,
                    "fund_name": r.fund_name,
                    "source_email_date": getattr(r, "source_email_date", None),
                    "raw_hash": getattr(r, "raw_hash", None),
                    "created_at": r.created_at,
                }
                for r in rows
            ]
        )
        .sort_values(["nav_date", "isin"])
    )

def _load_nav_summary(session) -> pd.DataFrame:
    rows = session.execute(
        select(
            FundNav.nav_date.label("nav_date"),
            func.count(FundNav.id).label("nav_count"),
        )
        .group_by(FundNav.nav_date)
        .order_by(FundNav.nav_date.desc())
    ).all()

    if not rows:
        return pd.DataFrame(columns=["nav_date", "nav_count"])

    return pd.DataFrame(rows, columns=["nav_date", "nav_count"])


def _load_nav_for_date(session, nav_date: date) -> pd.DataFrame:
    rows = (
        session.execute(
            select(FundNav)
            .where(FundNav.nav_date == nav_date)
            .order_by(FundNav.isin)
        )
        .scalars()
        .all()
    )

    if not rows:
        return pd.DataFrame(
            columns=["nav_date", "isin", "nav", "currency", "fund_name", "source_email_date", "raw_hash", "created_at"]
        )

    return pd.DataFrame(
        [
            {
                "nav_date": r.nav_date,
                "isin": r.isin,
                "nav": r.nav,
                "currency": r.currency,
                "fund_name": r.fund_name,
                "source_email_date": getattr(r, "source_email_date", None),
                "raw_hash": getattr(r, "raw_hash", None),
                "created_at": r.created_at,
            }
            for r in rows
        ]
    )

def _current_month_range() -> Tuple[date, date]:
    today = date.today()
    start = date(today.year, today.month, 1)
    end = date(today.year + 1, 1, 1) if today.month == 12 else date(today.year, today.month + 1, 1)
    return start, end

def _load_published(session, start: date, end: date) -> pd.DataFrame:
    rows = session.execute(
        select(PublishedRate)
        .where(PublishedRate.rate_date >= start)
        .where(PublishedRate.rate_date <= end)
    ).scalars().all()

    if not rows:
        return pd.DataFrame(columns=["rate_date", "series_code", "value"])

    return pd.DataFrame([{
        "rate_date": r.rate_date,
        "series_code": r.series_code,
        "value": r.value,
        "source": r.source,
    } for r in rows]).sort_values(["rate_date", "series_code"])

def get_published_watermark(session):
    # returns datetime.date or None
    return session.execute(select(func.max(PublishedRate.rate_date))).scalar_one()


def to_apollon_csv_bytes(out_df) -> bytes:
    """
    Output format (semicolon-separated):
    Fund Code;ValueDate;Rate;RateType
    ValueDate: DD.MM.YYYY
    RateType: always 'I'
    """
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["Fund Code", "ValueDate", "Rate", "RateType"])

    # Ensure stable ISIN order
    isins = list(APOLLON_FUND_CODE_BY_ISIN.keys())

    for ts, row in out_df.iterrows():
        d = ts.strftime("%d.%m.%Y")
        for isin in isins:
            fund_code = APOLLON_FUND_CODE_BY_ISIN[isin]
            val = row.get(isin)
            if val is None or (hasattr(val, "__float__") and (val != val)):
                rate = ""
            else:
                rate = f"{float(val):.{ROUND_DECIMALS}f}".replace(".", ",")
            w.writerow([fund_code, d, rate, "I"])

    return buf.getvalue().encode("utf-8")



# =============================================================================
# Language switcher (works even if only EN.json exists)
# =============================================================================

st.sidebar.markdown("---")
lang_label = t("lang", "Language")
lang_choice = st.sidebar.selectbox(
    lang_label,
    ["EN", "SK", "HU"],
    index=["EN", "SK", "HU"].index(st.session_state["lang"]) if st.session_state["lang"] in ["EN", "SK", "HU"] else 0,
)
if lang_choice != st.session_state["lang"]:
    st.session_state["lang"] = lang_choice
    _init_i18n()
    st.rerun()


# =============================================================================
# Main UI
# =============================================================================

render_header_with_logo(t("app_title", "Fund Rates Tool"))
st.caption(t("app_subtitle", "FX + NAV ingestion, audited calculation, XLS/CSV export"))

page_key = st.sidebar.radio(
    t("menu", "Menu"),
    ["menu_fx", "menu_nav", "menu_calc", "menu_audit", "menu_backfill"],
    format_func=lambda k: t(k, k),
)
    
with SessionLocal() as session:

    # -------------------------------------------------------------------------
    # 1) FX Import
    # -------------------------------------------------------------------------
    if page_key == "menu_fx":
        st.subheader(t("fx_title", "Import Tatra banka exchange rates from CSV (snapshot)"))
        st.caption(
            t(
                "fx_caption",
                "Upload TB 'Kurzový lístok' CSV. The file contains HUF/USD devíza buy/sell/mid but does not carry a date column, so you must assign the snapshot date.",
            )
        )

        uploaded_files = st.file_uploader(
            t("fx_upload", "Upload TB CSV (you may upload multiple files)"),
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.divider()
            st.markdown(f"### {t('fx_preview_save', 'Preview and save')}")

            for f in uploaded_files:
                auto_dt = try_parse_date_from_filename(f.name)
                chosen_dt = st.date_input(
                    f"{t('fx_rate_date_for', 'Rate date for')}: {f.name}",
                    value=auto_dt or date.today(),
                    key=f"fx_date_{f.name}",
                )

                try:
                    rates = parse_tb_rates_from_csv_bytes(f.getvalue())
                    preview_df = pd.DataFrame(
                        [
                            {
                                "rate_date": chosen_dt,
                                "huf_buy": rates["huf_buy"],
                                "huf_sell": rates["huf_sell"],
                                "huf_mid": rates["huf_mid"],
                                "usd_buy": rates["usd_buy"],
                                "usd_sell": rates["usd_sell"],
                                "usd_mid": rates["usd_mid"],
                            }
                        ]
                    )
                    # Show table (not JSON)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

                    save_label = f"{t('fx_save_btn', 'Save FX for')} {chosen_dt} ({f.name})"
                    if st.button(save_label, key=f"save_fx_{f.name}"):
                        obj = (
                            session.execute(select(FxRate).where(FxRate.rate_date == chosen_dt))
                            .scalar_one_or_none()
                        )
                        if obj is None:
                            obj = FxRate(rate_date=chosen_dt, **rates, source="csv_import")
                            session.add(obj)
                        else:
                            obj.huf_buy = rates["huf_buy"]
                            obj.huf_sell = rates["huf_sell"]
                            obj.huf_mid = rates["huf_mid"]
                            obj.usd_buy = rates["usd_buy"]
                            obj.usd_sell = rates["usd_sell"]
                            obj.usd_mid = rates["usd_mid"]
                            obj.source = "csv_import"
                        session.commit()
                        st.success(f"{t('fx_saved', 'Saved FX snapshot for')} {chosen_dt}.")

                except Exception as e:
                    st.error(f"{t('fx_parse_fail', 'Failed to parse')!s} '{f.name}': {e}")

        st.divider()
        st.markdown(f"### {t('fx_rows_db', 'FX rows stored in DB')}")
        fx_df = _load_fx(session)
        st.dataframe(fx_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # 2) NAV Paste
    # -------------------------------------------------------------------------
    elif page_key == "menu_nav":
        st.subheader(t("nav_title", "Paste NAV email (incl. header)"))
        st.caption(
            t(
                "nav_caption",
                "Paste the email including a header date with year (dd.mm.yyyy) or Outlook 'Sent:' line. NAV dates inside may be dd.mm.; January/December year rule is applied.",
            )
        )

        only_isins = set(NAV_CURRENCY.keys())

        pasted = st.text_area(t("nav_paste_label", "Paste email text here"), height=360)

        if "manual_nav_rows" not in st.session_state:
            st.session_state["manual_nav_rows"] = [
                {"isin": None, "nav_date": date.today(), "nav": None}
            ]

        
        # Parse
        if st.button(t("nav_parse_btn", "Parse NAVs")):
            try:
                df, email_dt = parse_baha_paste(pasted, only_isins=only_isins)
                st.info(
                    f"{t('nav_email_date_detected', 'Detected email date')}: {email_dt.strftime('%d.%m.%Y')}"
                )

                if df.empty:
                    st.warning(
                        t(
                            "nav_no_rows",
                            "No NAV rows detected. Ensure ISIN lines and NAV price lines are included.",
                        )
                    )
                    st.session_state.pop("parsed_nav_df", None)
                    st.session_state.pop("nav_email_dt", None)
                else:
                    st.session_state["parsed_nav_df"] = df
                    st.session_state["nav_email_dt"] = email_dt

            except Exception as e:
                st.error(str(e))
                st.session_state.pop("parsed_nav_df", None)
                st.session_state.pop("nav_email_dt", None)

        # Display / save (only when parsed df exists)
        if "parsed_nav_df" in st.session_state:
            df = st.session_state["parsed_nav_df"]
            email_dt = st.session_state.get("nav_email_dt", None)
            raw_hash = hashlib.sha256((pasted or "").encode("utf-8")).hexdigest()

            df_known = df[df["isin"].isin(only_isins)].copy()
            df_unknown = df[~df["isin"].isin(only_isins)].copy()

            if not df_unknown.empty:
                st.warning(
                    t("nav_unknown_isins", "Unknown ISINs (not in NAV_CURRENCY mapping):")
                    + f" {sorted(df_unknown['isin'].unique())}"
                )

            st.dataframe(df_known, use_container_width=True)

            if st.button(t("nav_save_btn", "Save parsed NAVs to DB")):
                saved = 0
                for _, r in df_known.iterrows():
                    obj = (
                        session.execute(
                            select(FundNav)
                            .where(FundNav.nav_date == r["nav_date"])
                            .where(FundNav.isin == r["isin"])
                        )
                        .scalar_one_or_none()
                    )

                    if obj is None:
                        obj = FundNav(
                            nav_date=r["nav_date"],
                            isin=r["isin"],
                            nav=float(r["nav"]),
                            currency=str(r["currency"]),
                            fund_name=r.get("fund_name", None),
                            source_email_date=email_dt,
                            raw_hash=raw_hash,
                        )
                        session.add(obj)
                    else:
                        obj.nav = float(r["nav"])
                        obj.currency = str(r["currency"])
                        obj.fund_name = r.get("fund_name", obj.fund_name)
                        obj.source_email_date = email_dt
                        obj.raw_hash = raw_hash

                    saved += 1

                session.commit()
                st.success(f"{t('nav_saved', 'Saved/updated NAV rows.')} ({saved})")
        
        st.divider()
        st.markdown(f"### {t('nav_manual_title', 'Manual NAV entry')}")

        with st.expander(t("nav_manual_exp", "Enter NAVs manually (when email is incomplete)"), expanded=True):
            # Build ISIN dropdown labels
            only_isins = sorted(set(NAV_CURRENCY.keys()))

            def _label(isin: str) -> str:
                nm = FUND_LABELS.get(isin, "")
                return f"{isin} — {nm}" if nm else isin

            isin_options = [""] + only_isins  # "" = not selected yet

            # Render each row
            for i, row in enumerate(st.session_state["manual_nav_rows"]):
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

                with c1:
                    current = row.get("isin") or ""
                    sel = st.selectbox(
                        t("nav_manual_isin", "ISIN – Fund name"),
                        options=isin_options,
                        index=isin_options.index(current) if current in isin_options else 0,
                        format_func=lambda x: "" if x == "" else _label(x),
                        key=f"manual_isin_{i}",
                    )
                    st.session_state["manual_nav_rows"][i]["isin"] = (sel or None)

                    # Show currency (read-only) once ISIN chosen
                    if sel:
                        st.caption(f"{t('nav_currency', 'Currency')}: {NAV_CURRENCY.get(sel, 'N/A')}")

                with c2:
                    d = st.date_input(
                        t("nav_manual_date", "NAV Date"),
                        value=row.get("nav_date") or date.today(),
                        key=f"manual_date_{i}",
                    )
                    st.session_state["manual_nav_rows"][i]["nav_date"] = d

                with c3:
                    v = st.number_input(
                        t("nav_manual_nav", "NAV"),
                        value=float(row["nav"]) if row.get("nav") not in (None, "") else 0.0,
                        min_value=0.0,
                        step=0.0001,
                        format="%.8f",
                        key=f"manual_nav_{i}",
                    )
                    # Treat 0.0 as "not provided" only if user left it untouched and ISIN empty; otherwise allow 0.0 if you want.
                    st.session_state["manual_nav_rows"][i]["nav"] = float(v)

                with c4:
                    if st.button(t("nav_manual_remove", "Remove"), key=f"manual_remove_{i}"):
                        st.session_state["manual_nav_rows"].pop(i)
                        st.rerun()

            col_add, col_save = st.columns([1, 2])
            with col_add:
                if st.button(t("nav_manual_add", "Add new NAV")):
                    st.session_state["manual_nav_rows"].append({"isin": None, "nav_date": date.today(), "nav": None})
                    st.rerun()

            with col_save:
                if st.button(t("nav_manual_save", "Save NAVs")):
                    rows = st.session_state["manual_nav_rows"]

                    # Basic validation
                    errors = []
                    for idx, r in enumerate(rows, start=1):
                        if not r.get("isin"):
                            errors.append(t("nav_err_isin_missing", "Row {row}: ISIN is missing.").format(row=idx))
                        if not r.get("nav_date"):
                            errors.append(t("nav_err_date_missing", "Row {row}: NAV date is missing.").format(row=idx))
                        # nav must be > 0 in typical NAV logic; adjust if 0 allowed
                        if r.get("nav") is None or float(r.get("nav")) <= 0:
                            errors.append(t("nav_err_nav_leq0", "Row {row}: NAV must be > 0.").format(row=idx))

                        if r.get("isin") and r["isin"] not in NAV_CURRENCY:
                            errors.append(t("nav_err_isin_unknown", "Row {row}: ISIN not recognized in NAV_CURRENCY.").format(row=idx))

                    if errors:
                        st.error(t("nav_manual_errors", "Please fix:") + "\n- " + "\n- ".join(errors))
                    else:
                        saved = 0
                        # Hash to trace manual payload (optional)
                        payload_str = json.dumps(rows, sort_keys=True, default=str)
                        raw_hash = hashlib.sha256(("manual:" + payload_str).encode("utf-8")).hexdigest()

                        for r in rows:
                            isin = r["isin"]
                            nav_date = r["nav_date"]
                            nav_val = float(r["nav"])
                            currency = str(NAV_CURRENCY.get(isin, ""))  # from config
                            fund_name = FUND_LABELS.get(isin)
        
                            obj = (
                                session.execute(
                                    select(FundNav)
                                    .where(FundNav.nav_date == nav_date)
                                    .where(FundNav.isin == isin)
                                )
                                .scalar_one_or_none()
                            )

                            if obj is None:
                                obj = FundNav(
                                    nav_date=nav_date,
                                    isin=isin,
                                    nav=nav_val,
                                    currency=currency,
                                    fund_name=fund_name,
                                    source_email_date=None,
                                    raw_hash=raw_hash,
                                )
                                session.add(obj)
                            else:
                                obj.nav = nav_val
                                obj.currency = currency
                                if fund_name:
                                    obj.fund_name = fund_name
                                obj.source_email_date = None
                                obj.raw_hash = raw_hash

                            saved += 1

                        session.commit()
                        st.success(f"{t('nav_manual_saved', 'Saved/updated NAV rows.')} ({saved})")

                        # Optionally reset to a single empty row
                        st.session_state["manual_nav_rows"] = [{"isin": None, "nav_date": date.today(), "nav": None}]
                        st.rerun()
        
        st.divider()
        st.markdown(f"### {t('nav_rows_db', 'NAV rows stored in DB')}")

        nav_summary_df = _load_nav_summary(session)

        if nav_summary_df.empty:
            st.info(t("nav_rows_db_empty", "No NAV rows stored in DB yet."))
        else:
             expand_all = st.checkbox(t("nav_expand_all", "Expand all days"), value=False)

             for _, r in nav_summary_df.iterrows():
                 d = r["nav_date"]
                 cnt = int(r["nav_count"])

                 with st.expander(f"{d} ({cnt})", expanded=expand_all):
                     day_df = _load_nav_for_date(session, d)
                     st.dataframe(day_df, use_container_width=True, hide_index=True)


    # -------------------------------------------------------------------------
    # 3) Cash allocation
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # 4) Calculate & Export
    # -------------------------------------------------------------------------
    elif page_key == "menu_calc":
        st.subheader(t("calc_title", "Calculate outputs and export"))

        fx_df = _load_fx(session)
        nav_df = _load_nav(session)

        wm = get_published_watermark(session)
        if wm:
            st.info(f"{t('calc_watermark', 'Published history watermark')}: {wm}")
        else:
            st.info(t("calc_watermark_none", "No published history watermark found yet."))

        import datetime as dt

        default_from = (wm + dt.timedelta(days=1)) if wm else _current_month_range()[0]
        default_to = date.today()

        colA, colB, colC = st.columns(3)
        with colA:
            tr_yield = st.number_input(
                t("calc_tr_yield", "TR yearly yield"),
                value=float(TR_YEARLY_YIELD_DEFAULT),
                step=0.001,
                format="%.6f",
            )
        with colB:
            require_all_navs = st.checkbox(
                t("calc_require_all_navs", "Require all NAVs for a date"),
                value=True,
            )
        with colC:
            require_fx_same_day = st.checkbox(
                t("calc_require_fx_same_day", "Require FX same-day (no carry-forward)"),
                value=False,
            )

        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(t("calc_from", "From"), value=default_from)
        with col2:
            date_to = st.date_input(t("calc_to", "To (inclusive)"), value=default_to)

        # Deterministic anchor: the day before the requested start date.
        # The engine will run only if published/backfilled history contains ALL series on this anchor date.
        anchor_required = date_from - dt.timedelta(days=1)

        # Anchor cutoff (<= D-1): the engine will choose the latest published day on/before this cutoff
        # where all series exist (handles weekends/holidays). If none exists, it will compute nothing.
        st.info(
            f"{t('calc_anchor', 'Anchor cutoff (<= D-1)')}: {anchor_required:%Y-%m-%d}. "
            f"{t('calc_anchor_rule', 'The engine will use the latest published date on or before this cutoff where all series exist; otherwise nothing will be calculated.')}"
        )

        if st.button(t("calc_run", "Run calculation")):
            # Load a small window of published history up to the cutoff so compute_outputs can pick the best anchor.
            pub_start = anchor_required - dt.timedelta(days=60)
            published_long = _load_published(session, pub_start, anchor_required)

            out_df, meta, coverage = compute_outputs(
                fx_df=fx_df,
                nav_df=nav_df,
                tr_yearly_yield=tr_yield,
                require_all_navs=require_all_navs,
                require_fx_same_day=require_fx_same_day,
                use_watermark_anchor=False,  # legacy param (unused in strict anchor mode)
                published_df_long=published_long,
                date_from=date_from,   # datetime.date
                date_to=date_to,       # datetime.date
            )

            st.session_state["computed_out_df"] = out_df.copy()

            if coverage.get("no_anchor"):
                st.error(
                    t(
                        "calc_no_anchor",
                        "Strict mode: no anchor date found. Import NAV+FX for at least one date that exists in published history for all series.",
                     )
                )
                diag = {
                    k: coverage.get(k)
                    for k in ["published_max_date", "anchor_candidates", "missing_anchor_series"]
                    if k in coverage
                }
                if diag:
                    st.json(diag)
                st.session_state.pop("out_df", None)
                st.session_state.pop("out_meta", None)
                st.session_state.pop("out_coverage", None)
                st.stop()
                
            if coverage.get("nav_dates_missing_fx"):
                st.warning(t("calc_missing_fx_warn", "Some NAV dates have no FX yet."))
                st.dataframe(
                    pd.DataFrame({"nav_date_missing_fx": coverage["nav_dates_missing_fx"]}),
                    use_container_width=True,
                )

            if coverage.get("nav_dates_incomplete"):
                st.warning(
                    t(
                        "calc_incomplete_nav_warn",
                        "Some NAV dates are incomplete (missing one or more fund NAVs).",
                    )
                )
                st.dataframe(
                    pd.DataFrame({"nav_date_incomplete": coverage["nav_dates_incomplete"]}),
                    use_container_width=True,
                )

            if out_df.empty:
                st.error(
                    t(
                        "calc_no_valid",
                        "No valid output dates yet. Import the missing FX or NAVs to unlock calculation.",
                    )
                )
                st.session_state.pop("out_df", None)
                st.session_state.pop("out_meta", None)
                st.session_state.pop("out_coverage", None)
                st.stop()

            # Published history is used ONLY for anchoring (D-1), not for overwriting computed values.
            final_out = out_df.reindex(columns=SERIES_ORDER)

            out_f = final_out.loc[
                (final_out.index.date >= date_from) & (final_out.index.date <= date_to)
                ].copy()

            st.session_state["out_df"] = out_f
            st.session_state["out_meta"] = meta
            st.session_state["out_coverage"] = coverage

            st.success(t("calc_complete", "Calculation complete."))
            st.dataframe(
                out_f.reset_index().rename(columns={"index": t("date", "Date")}),
                use_container_width=True,
            )

        if "out_df" in st.session_state:
            out_f = st.session_state["out_df"]
            meta = st.session_state.get("out_meta", {})
            coverage = st.session_state.get("out_coverage", {})

            st.subheader(t("export_title", "Export"))
            
            # Apollon filename: FundRates_HUFunds_DATEFROM-DATETO.csv (DDMMYY)
            d_from = out_f.index.min().date()
            d_to = out_f.index.max().date()
            apollon_name = f"FundRates_HUFunds_{d_from:%d%m%y}-{d_to:%d%m%y}.csv"

            st.download_button(
                t("export_apollon_csv", "Download CSV for Apollon"),
                data=to_apollon_csv_bytes(out_f),
                file_name=apollon_name,
                mime="text/csv",
            )
            st.download_button(
                t("export_csv", "Download CSV"),
                data=to_csv_bytes(out_f),
                file_name="kurzy_export.csv",
                mime="text/csv",
            )
            st.download_button(
                t("export_xlsx", "Download XLSX"),
                data=to_xlsx_bytes(out_f),
                file_name="kurzy_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader(t("audit_save_title", "Persist this run (audit)"))
            if st.button(t("audit_save_btn", "Save calculation run to DB")):
                payload = {
                    "meta": meta,
                    "filter_from": str(date_from),
                    "filter_to": str(date_to),
                    "fx_rows": int(len(fx_df)),
                    "nav_rows": int(len(nav_df)),
                    "valid_dates": coverage.get("valid_dates", []),
                }
                h = compute_input_hash(payload)

                run = CalcRun(params_json=json.dumps(payload, sort_keys=True), input_hash=h)
                session.add(run)
                session.flush()

                for idx, row in out_f.iterrows():
                    session.add(
                        CalcDaily(
                            run_id=run.id,
                            calc_date=idx.date(),
                            output_json=json.dumps(
                                {k: float(row[k]) for k in SERIES_ORDER if k in out_f.columns},
                                sort_keys=True,
                            ),
                        )
                    )

                session.commit()
                st.success(
                    f"{t('audit_saved', 'Saved calc_run')} id={run.id} ({len(out_f)} {t('rows', 'rows')})."
                )

            st.subheader(t("calc_publish_title", "Append computed days to published history"))
            if st.button(
                t(
                    "calc_publish_btn",
                    "Save computed outputs after watermark to published history",
                )
            ):
                df_to_save = out_df.copy()
                df_to_save["rate_date"] = pd.to_datetime(df_to_save.index).date
                if wm:
                    df_to_save = df_to_save[df_to_save["rate_date"] > wm]

                if df_to_save.empty:
                    st.info(
                        t(
                            "calc_publish_none",
                            "Nothing new to save (all computed dates are <= watermark).",
                        )
                    )
                else:
                    long = df_to_save.drop(columns=["rate_date"]).copy()
                    long["rate_date"] = df_to_save["rate_date"].values
                    long = (
                        long.melt(
                            id_vars=["rate_date"],
                            var_name="series_code",
                            value_name="value",
                        )
                        .dropna()
                    )

                    n = upsert_published_rates(
                        session,
                        long[["rate_date", "series_code", "value"]],
                        source="computed",
                    )
                    session.commit()
                    st.success(
                        t("calc_publish_saved", "Saved computed rows to published history:")
                        + f" {n:,}"
                    )

    # -------------------------------------------------------------------------
    # 5) Audit runs
    # -------------------------------------------------------------------------
    elif page_key == "menu_audit":
        st.subheader(t("audit_page_title", "Audit: saved calculation runs"))

        runs = session.execute(select(CalcRun).order_by(CalcRun.created_at.desc())).scalars().all()
        if not runs:
            st.info(t("audit_no_runs", "No saved calc runs yet."))
        else:
            run_opts = {
                f"{t('audit_run', 'Run')} {r.id} @ {r.created_at.isoformat()} ({t('audit_hash', 'hash')} {r.input_hash[:10]}…)": r.id
                for r in runs
            }
            choice = st.selectbox(t("audit_select_run", "Select run"), list(run_opts.keys()))
            run_id = run_opts[choice]

            run = session.execute(select(CalcRun).where(CalcRun.id == run_id)).scalar_one()
            st.code(run.params_json, language="json")

            rows = (
                session.execute(
                    select(CalcDaily)
                    .where(CalcDaily.run_id == run_id)
                    .order_by(CalcDaily.calc_date)
                )
                .scalars()
                .all()
            )

            out = []
            for r in rows:
                d = json.loads(r.output_json)
                date_col = t("date", "Date")
                d[date_col] = r.calc_date
                out.append(d)

            df = pd.DataFrame(out).sort_values(date_col) if out else pd.DataFrame()
            st.dataframe(df, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5) Backfill
    # -------------------------------------------------------------------------
        
    elif page_key == "menu_backfill":
        st.subheader(t("backfill_title", "Backfill published outputs from legacy XLSM"))

        x = st.file_uploader(
            t("backfill_uploader", "Upload the legacy XLSM backfill file"),
            accept_multiple_files=False,
        )
        if x:
            df_long = backfill_legacy.parse_legacy_outputs_xlsm(x.getvalue())
            st.write(f"{t('rows', 'Rows')}: {len(df_long):,}")
            st.dataframe(df_long.head(50), use_container_width=True)

            if st.button(t("backfill_import_btn", "Import / Upsert into DB")):
                n = upsert_published_rates(session, df_long, source="xlsm_backfill")
                session.commit()
                st.success(
                    f"{t('backfill_upserted', 'Upserted rows into published_rates')}: {n:,}"
                )
































