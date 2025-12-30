from __future__ import annotations

import base64
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import select

from db import (
    SessionLocal,
    init_db,
    FxRate,
    FundNav,
    CashAlloc,
    CalcRun,
    CalcDaily,
    compute_input_hash,
)
from fx_import import parse_tb_rates_from_csv_bytes, try_parse_date_from_filename
from nav_parser import parse_baha_paste
from calc import compute_outputs
from export import to_csv_bytes, to_xlsx_bytes
from config import NAV_CURRENCY, SERIES_ORDER, TR_YEARLY_YIELD_DEFAULT


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
        return pd.DataFrame(columns=["nav_date", "isin", "nav", "currency", "fund_name"])
    return (
        pd.DataFrame(
            [
                {
                    "nav_date": r.nav_date,
                    "isin": r.isin,
                    "nav": r.nav,
                    "currency": r.currency,
                    "fund_name": r.fund_name,
                    "source": r.source,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
        )
        .sort_values(["nav_date", "isin"])
    )


def _load_cash(session) -> pd.DataFrame:
    rows = session.execute(select(CashAlloc)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["alloc_date", "series_code", "cash_pct"])
    return (
        pd.DataFrame(
            [
                {
                    "alloc_date": r.alloc_date,
                    "series_code": r.series_code,
                    "cash_pct": r.cash_pct,
                }
                for r in rows
            ]
        )
        .sort_values(["alloc_date", "series_code"])
    )


def _current_month_range() -> Tuple[date, date]:
    today = date.today()
    start = date(today.year, today.month, 1)
    end = date(today.year + 1, 1, 1) if today.month == 12 else date(today.year, today.month + 1, 1)
    return start, end


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

page = st.sidebar.radio(
    t("menu", "Menu"),
    [
        t("menu_fx", "1) Import FX (TB CSV)"),
        t("menu_nav", "2) Paste NAV email"),
        t("menu_cash", "3) Cash allocation"),
        t("menu_calc", "4) Calculate & Export"),
        t("menu_audit", "5) Audit runs"),
    ],
)

with SessionLocal() as session:

    # -------------------------------------------------------------------------
    # 1) FX Import
    # -------------------------------------------------------------------------
    if page == t("menu_fx", "1) Import FX (TB CSV)"):
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
    elif page == t("menu_nav", "2) Paste NAV email"):
        st.subheader(t("nav_title", "Paste NAV email (incl. header)"))
        st.caption(
            t(
                "nav_caption",
                "Paste the email including a header date with year (dd.mm.yyyy) or Outlook 'Sent:' line. NAV dates inside may be dd.mm.; January/December year rule is applied.",
            )
        )

        only_isins = set(NAV_CURRENCY.keys())

        pasted = st.text_area(t("nav_paste_label", "Paste email text here"), height=360)

        # Parse
        if st.button(t("nav_parse_btn", "Parse NAVs")):
            try:
                df, email_dt = parse_baha_paste(pasted, only_isins=only_isins)  # FIX: was `none`
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
                else:
                    st.session_state["parsed_nav_df"] = df

            except Exception as e:
                st.error(str(e))
                st.session_state.pop("parsed_nav_df", None)

        # Display / save (only when parsed df exists)
        if "parsed_nav_df" in st.session_state:
            df = st.session_state["parsed_nav_df"]

            df_known = df[df["isin"].isin(only_isins)].copy()
            df_unknown = df[~df["isin"].isin(only_isins)].copy()

            if not df_unknown.empty:
                st.warning(
                    f"Unknown ISINs (not in NAV_CURRENCY mapping): {sorted(df_unknown['isin'].unique())}"
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
                            raw_excerpt=r.get("raw_excerpt", None),
                            source="paste",
                        )
                        session.add(obj)
                    else:
                        obj.nav = float(r["nav"])
                        obj.currency = str(r["currency"])
                        obj.fund_name = r.get("fund_name", obj.fund_name)
                        obj.raw_excerpt = r.get("raw_excerpt", obj.raw_excerpt)

                    saved += 1

                session.commit()
                st.success(f"{t('nav_saved', 'Saved/updated NAV rows.')} ({saved})")

        st.divider()
        st.markdown(f"### {t('nav_rows_db', 'NAV rows stored in DB')}")
        nav_df = _load_nav(session)
        st.dataframe(nav_df, use_container_width=True)


    # -------------------------------------------------------------------------
    # 3) Cash allocation
    # -------------------------------------------------------------------------
    elif page == t("menu_cash", "3) Cash allocation"):
        st.subheader(t("cash_title", "Cash allocation (0..1) per series per date (optional)"))
        st.caption(
            t(
                "cash_caption",
                "Matches Excel delta-damping: out_t = out_{t-1} + (base_t - base_{t-1})*(1-cash_pct). Leave at 0 if not used.",
            )
        )

        nav_df = _load_nav(session)
        if nav_df.empty:
            st.warning(t("cash_enter_nav_first", "Enter NAVs first."))
        else:
            cash_df = _load_cash(session)
            nav_dates = sorted(pd.to_datetime(nav_df["nav_date"]).dt.date.unique().tolist())

            wide = cash_df.pivot_table(index="alloc_date", columns="series_code", values="cash_pct", aggfunc="last")
            wide = wide.reindex(nav_dates)

            for c in SERIES_ORDER:
                if c not in wide.columns:
                    wide[c] = 0.0
            wide = wide[SERIES_ORDER].fillna(0.0)
            wide.index.name = "alloc_date"

            edited = st.data_editor(wide.reset_index(), use_container_width=True, num_rows="fixed")

            if st.button(t("cash_save_btn", "Save cash allocation")):
                saved = 0
                for _, r in edited.iterrows():
                    d = r["alloc_date"]
                    if pd.isna(d):
                        continue
                    for code in SERIES_ORDER:
                        v = r.get(code, 0.0)
                        if pd.isna(v):
                            v = 0.0
                        v = float(v)
                        if v < 0 or v > 1:
                            st.error(
                                t("cash_range_err", "Cash % must be 0..1.")
                                + f" {t('cash_range_detail', 'Found')} {v} {t('cash_range_for', 'for')} {code} {t('cash_range_on', 'on')} {d}."
                            )
                            st.stop()

                        obj = (
                            session.execute(
                                select(CashAlloc)
                                .where(CashAlloc.alloc_date == d)
                                .where(CashAlloc.series_code == code)
                            )
                            .scalar_one_or_none()
                        )

                        if obj is None:
                            session.add(CashAlloc(alloc_date=d, series_code=code, cash_pct=v, source="manual"))
                        else:
                            obj.cash_pct = v
                        saved += 1

                session.commit()
                st.success(f"{t('cash_saved', 'Saved/updated cash cells.')} ({saved})")

    # -------------------------------------------------------------------------
    # 4) Calculate & Export
    # -------------------------------------------------------------------------
    elif page == t("menu_calc", "4) Calculate & Export"):
        st.subheader(t("calc_title", "Calculate outputs and export"))

        fx_df = _load_fx(session)
        nav_df = _load_nav(session)
        cash_df = _load_cash(session)

        colA, colB, colC = st.columns(3)
        with colA:
            tr_yield = st.number_input(
                t("calc_tr_yield", "TR yearly yield"),
                value=float(TR_YEARLY_YIELD_DEFAULT),
                step=0.001,
                format="%.6f",
            )
        with colB:
            require_all_navs = st.checkbox(t("calc_require_all_navs", "Require all 9 NAVs for a date"), value=True)
        with colC:
            require_fx_same_day = st.checkbox(
                t("calc_require_fx_same_day", "Require FX same-day (no carry-forward)"),
                value=False,
            )

        start_default, _ = _current_month_range()
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(t("calc_from", "From"), value=start_default)
        with col2:
            date_to = st.date_input(t("calc_to", "To (inclusive)"), value=date.today())

        if st.button(t("calc_run", "Run calculation")):
            out_df, meta, coverage = compute_outputs(
                fx_df=fx_df,
                nav_df=nav_df,
                cash_df=cash_df,
                tr_yearly_yield=float(tr_yield),
                require_all_navs=require_all_navs,
                require_fx_same_day=require_fx_same_day,
            )

            st.subheader(t("calc_diag_title", "Coverage diagnostics"))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(t("calc_fx_rows", "FX rows in DB"), 0 if fx_df.empty else len(fx_df))
            with c2:
                st.metric(t("calc_nav_rows", "NAV rows in DB"), 0 if nav_df.empty else len(nav_df))
            with c3:
                st.metric(t("calc_valid_dates", "Valid output dates"), len(coverage.get("valid_dates", [])))

            if coverage.get("nav_dates_missing_fx"):
                st.warning(t("calc_missing_fx_warn", "Some NAV dates currently cannot be calculated (missing FX coverage)."))
                st.dataframe(
                    pd.DataFrame({"nav_date_missing_fx": coverage["nav_dates_missing_fx"]}),
                    use_container_width=True,
                )

            if coverage.get("nav_dates_incomplete"):
                st.warning(t("calc_incomplete_nav_warn", "Some NAV dates are incomplete (missing one or more fund NAVs)."))
                st.dataframe(
                    pd.DataFrame({"nav_date_incomplete": coverage["nav_dates_incomplete"]}),
                    use_container_width=True,
                )

            if out_df.empty:
                st.error(t("calc_no_valid", "No valid output dates yet. Import the missing FX or NAVs to unlock calculation."))
                st.stop()

            out_f = out_df.loc[(out_df.index.date >= date_from) & (out_df.index.date <= date_to)].copy()
            st.session_state["out_df"] = out_f
            st.session_state["out_meta"] = meta
            st.session_state["out_coverage"] = coverage

            st.success(t("calc_complete", "Calculation complete."))
            st.dataframe(out_f.reset_index().rename(columns={"index": "Date"}), use_container_width=True)

        if "out_df" in st.session_state:
            out_f = st.session_state["out_df"]
            meta = st.session_state.get("out_meta", {})
            coverage = st.session_state.get("out_coverage", {})

            st.subheader(t("export_title", "Export"))
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
                    "cash_rows": int(len(cash_df)),
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
                            output_json=json.dumps({k: float(row[k]) for k in SERIES_ORDER}, sort_keys=True),
                        )
                    )

                session.commit()
                st.success(f"{t('audit_saved', 'Saved calc_run')} id={run.id} ({len(out_f)} {t('rows', 'rows')}).")

    # -------------------------------------------------------------------------
    # 5) Audit runs
    # -------------------------------------------------------------------------
    elif page == t("menu_audit", "5) Audit runs"):
        st.subheader(t("audit_page_title", "Audit: saved calculation runs"))

        runs = session.execute(select(CalcRun).order_by(CalcRun.run_ts.desc())).scalars().all()
        if not runs:
            st.info(t("audit_no_runs", "No saved calc runs yet."))
        else:
            run_opts = {
                f"{t('audit_run', 'Run')} {r.id} @ {r.run_ts.isoformat()} ({t('audit_hash', 'hash')} {r.input_hash[:10]}…)": r.id
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
                d["Date"] = r.calc_date
                out.append(d)

            df = pd.DataFrame(out).sort_values("Date") if out else pd.DataFrame()
            st.dataframe(df, use_container_width=True)



