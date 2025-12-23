from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import streamlit as st
from sqlalchemy import select

from db import SessionLocal, init_db, FxRate, FundNav, CashAlloc, CalcRun, CalcDaily, compute_input_hash
from fx_import import parse_tb_rates_from_csv_bytes, try_parse_date_from_filename
from nav_parser import parse_baha_paste
from calc import compute_outputs
from export import to_csv_bytes, to_xlsx_bytes
from config import NAV_CURRENCY, SERIES_ORDER, TR_YEARLY_YIELD_DEFAULT
from pathlib import Path
import base64

BASE_DIR = Path(__file__).parent
LOGO_BIG = BASE_DIR / "assets/FundRatesTool_logo.png"
LOGO_SMALL = BASE_DIR / "assets/FundRatesTool_logo_small.png"

def render_header_with_logo(title_text: str):
    """Render top header with logo and title aligned in one row."""
    if not LOGO_BIG.exists():
        st.title(title_text)
        return

    with LOGO_BIG.open("rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")

    html = f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    ">
      <img src="data:image/png;base64,{b64}"
           style="height: 125px; width: auto; display: block;" />
      <h1 style="margin: 0; font-size: 2.4rem; font-weight: 700; color: #2F3136;">
        {title_text}
      </h1>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(
    page_title="Pension Options Tool",
    page_icon=str(LOGO_SMALL),   # small logo in browser tab
    layout="wide",
)

@st.cache_resource
def _init():
    init_db()
    return True


_init()


def _load_fx(session) -> pd.DataFrame:
    rows = session.execute(select(FxRate)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["rate_date","huf_buy","huf_sell","huf_mid","usd_buy","usd_sell","usd_mid"])
    return pd.DataFrame([{
        "rate_date": r.rate_date,
        "huf_buy": r.huf_buy, "huf_sell": r.huf_sell, "huf_mid": r.huf_mid,
        "usd_buy": r.usd_buy, "usd_sell": r.usd_sell, "usd_mid": r.usd_mid,
        "source": r.source,
        "created_at": r.created_at,
    } for r in rows]).sort_values("rate_date")


def _load_nav(session) -> pd.DataFrame:
    rows = session.execute(select(FundNav)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["nav_date","isin","nav","currency","fund_name"])
    return pd.DataFrame([{
        "nav_date": r.nav_date,
        "isin": r.isin,
        "nav": r.nav,
        "currency": r.currency,
        "fund_name": r.fund_name,
        "source": r.source,
        "created_at": r.created_at,
    } for r in rows]).sort_values(["nav_date","isin"])


def _load_cash(session) -> pd.DataFrame:
    rows = session.execute(select(CashAlloc)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["alloc_date","series_code","cash_pct"])
    return pd.DataFrame([{
        "alloc_date": r.alloc_date,
        "series_code": r.series_code,
        "cash_pct": r.cash_pct,
    } for r in rows]).sort_values(["alloc_date","series_code"])


def _current_month_range() -> tuple[date, date]:
    today = date.today()
    start = date(today.year, today.month, 1)
    end = date(today.year + 1, 1, 1) if today.month == 12 else date(today.year, today.month + 1, 1)
    return start, end

render_header_with_logo(t["app_title"])

page = st.sidebar.radio(
    "Menu",
    ["1) Import FX (TB CSV)", "2) Paste NAV email", "3) Cash allocation", "4) Calculate & Export", "5) Audit runs"]
)

with SessionLocal() as session:

    if page == "1) Import FX (TB CSV)":
        st.subheader("Import Tatra banka exchange rates from CSV (snapshot)")
        st.caption(
            "Upload TB 'Kurzový lístok' CSV. The file contains HUF/USD devíza buy/sell/mid but does not carry a date column, "
            "so you must assign the snapshot date."
        )

        uploaded_files = st.file_uploader(
            "Upload TB CSV (you may upload multiple files)",
            type=["csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.divider()
            st.write("### Preview and save")

            for f in uploaded_files:
                auto_dt = try_parse_date_from_filename(f.name)
                chosen_dt = st.date_input(
                    f"Rate date for: {f.name}",
                    value=auto_dt or date.today(),
                    key=f"fx_date_{f.name}"
                )

                try:
                    rates = parse_tb_rates_from_csv_bytes(f.getvalue())
                    st.json({"rate_date": str(chosen_dt), **rates})

                    if st.button(f"Save FX for {chosen_dt} ({f.name})", key=f"save_fx_{f.name}"):
                        obj = session.execute(select(FxRate).where(FxRate.rate_date == chosen_dt)).scalar_one_or_none()
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
                        st.success(f"Saved FX snapshot for {chosen_dt}.")

                except Exception as e:
                    st.error(f"Failed to parse '{f.name}': {e}")

        st.divider()
        st.write("### FX rows stored in DB")
        fx_df = _load_fx(session)
        st.dataframe(fx_df, use_container_width=True)

    elif page == "2) Paste NAV email":
        st.subheader("Paste NAV email (incl. header)")
        st.caption(
            "Paste the full Baha email including a header date with year (dd.mm.yyyy). "
            "NAV dates inside can be dd.mm. without year; January/December year rule is applied."
        )

        only_isins = set(NAV_CURRENCY.keys())
        pasted = st.text_area("Paste email text here", height=360)

        if st.button("Parse NAVs"):
            try:
                df, email_dt = parse_baha_paste(pasted, only_isins=only_isins)
                st.info(f"Detected email date: {email_dt.strftime('%d.%m.%Y')}")
                if df.empty:
                    st.warning("No NAV rows detected. Ensure ISIN lines and 'Last Curr.' price lines are included.")
                else:
                    st.dataframe(df, use_container_width=True)
                    st.session_state["parsed_nav_df"] = df
            except Exception as e:
                st.error(str(e))

        if "parsed_nav_df" in st.session_state:
            if st.button("Save parsed NAVs to DB"):
                df = st.session_state["parsed_nav_df"]
                saved = 0
                for _, r in df.iterrows():
                    obj = session.execute(
                        select(FundNav).where(FundNav.nav_date == r["nav_date"]).where(FundNav.isin == r["isin"])
                    ).scalar_one_or_none()

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
                st.success(f"Saved/updated {saved} NAV rows.")

        st.divider()
        st.write("### NAV rows stored in DB")
        nav_df = _load_nav(session)
        st.dataframe(nav_df, use_container_width=True)

    elif page == "3) Cash allocation":
        st.subheader("Cash allocation (0..1) per series per date (optional)")
        st.caption(
            "This matches your Excel delta-damping: out_t = out_{t-1} + (base_t - base_{t-1})*(1-cash_pct). "
            "Leave at 0 if not used."
        )

        nav_df = _load_nav(session)
        if nav_df.empty:
            st.warning("Enter NAVs first.")
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

            if st.button("Save cash allocation"):
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
                            st.error(f"Cash % must be 0..1. Found {v} for {code} on {d}.")
                            st.stop()

                        obj = session.execute(
                            select(CashAlloc).where(CashAlloc.alloc_date == d).where(CashAlloc.series_code == code)
                        ).scalar_one_or_none()

                        if obj is None:
                            session.add(CashAlloc(alloc_date=d, series_code=code, cash_pct=v, source="manual"))
                        else:
                            obj.cash_pct = v
                        saved += 1

                session.commit()
                st.success(f"Saved/updated {saved} cash cells.")

    elif page == "4) Calculate & Export":
        st.subheader("Calculate outputs and export")
        fx_df = _load_fx(session)
        nav_df = _load_nav(session)
        cash_df = _load_cash(session)

        colA, colB, colC = st.columns(3)
        with colA:
            tr_yield = st.number_input("TR yearly yield", value=float(TR_YEARLY_YIELD_DEFAULT), step=0.001, format="%.6f")
        with colB:
            require_all_navs = st.checkbox("Require all 9 NAVs for a date", value=True)
        with colC:
            require_fx_same_day = st.checkbox("Require FX same-day (no carry-forward)", value=False)

        start_default, _ = _current_month_range()
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From", value=start_default)
        with col2:
            date_to = st.date_input("To (inclusive)", value=date.today())

        if st.button("Run calculation"):
            out_df, meta, coverage = compute_outputs(
                fx_df=fx_df,
                nav_df=nav_df,
                cash_df=cash_df,
                tr_yearly_yield=float(tr_yield),
                require_all_navs=require_all_navs,
                require_fx_same_day=require_fx_same_day,
            )

            st.subheader("Coverage diagnostics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("FX rows in DB", 0 if fx_df.empty else len(fx_df))
            with c2:
                st.metric("NAV rows in DB", 0 if nav_df.empty else len(nav_df))
            with c3:
                st.metric("Valid output dates", len(coverage["valid_dates"]))

            if coverage["nav_dates_missing_fx"]:
                st.warning("Some NAV dates currently cannot be calculated (missing FX coverage).")
                st.dataframe(pd.DataFrame({"nav_date_missing_fx": coverage["nav_dates_missing_fx"]}), use_container_width=True)

            if coverage["nav_dates_incomplete"]:
                st.warning("Some NAV dates are incomplete (missing one or more fund NAVs).")
                st.dataframe(pd.DataFrame({"nav_date_incomplete": coverage["nav_dates_incomplete"]}), use_container_width=True)

            if out_df.empty:
                st.error("No valid output dates yet. Import the missing FX or NAVs to unlock calculation.")
                st.stop()

            out_f = out_df.loc[(out_df.index.date >= date_from) & (out_df.index.date <= date_to)].copy()
            st.session_state["out_df"] = out_f
            st.session_state["out_meta"] = meta
            st.session_state["out_coverage"] = coverage

            st.success("Calculation complete.")
            st.dataframe(out_f.reset_index().rename(columns={"index": "Date"}), use_container_width=True)

        if "out_df" in st.session_state:
            out_f = st.session_state["out_df"]
            meta = st.session_state.get("out_meta", {})
            coverage = st.session_state.get("out_coverage", {})

            st.subheader("Export")
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(out_f),
                file_name="kurzy_export.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download XLSX",
                data=to_xlsx_bytes(out_f),
                file_name="kurzy_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader("Persist this run (audit)")
            if st.button("Save calculation run to DB"):
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
                st.success(f"Saved calc_run id={run.id} with {len(out_f)} rows.")

    elif page == "5) Audit runs":
        st.subheader("Audit: saved calculation runs")

        runs = session.execute(select(CalcRun).order_by(CalcRun.run_ts.desc())).scalars().all()
        if not runs:
            st.info("No saved calc runs yet.")
        else:
            run_opts = {f"Run {r.id} @ {r.run_ts.isoformat()} (hash {r.input_hash[:10]}…)": r.id for r in runs}
            choice = st.selectbox("Select run", list(run_opts.keys()))
            run_id = run_opts[choice]

            run = session.execute(select(CalcRun).where(CalcRun.id == run_id)).scalar_one()
            st.code(run.params_json, language="json")

            rows = session.execute(select(CalcDaily).where(CalcDaily.run_id == run_id).order_by(CalcDaily.calc_date)).scalars().all()
            out = []
            for r in rows:
                d = json.loads(r.output_json)
                d["Date"] = r.calc_date
                out.append(d)

            df = pd.DataFrame(out).sort_values("Date")
            st.dataframe(df, use_container_width=True)





