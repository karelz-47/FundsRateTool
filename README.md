# Fund Rates Calculator (Streamlit)

## What it does
- Import Tatra banka FX (HUF/USD devíza buy/sell/mid) from CSV snapshots (user assigns date).
- Paste NAV email (incl header with dd.mm.yyyy). NAV dates inside may be dd.mm.; January/December year rule applied.
- Store FX, NAV, and optional cash allocation in Postgres (Railway DATABASE_URL).
- Calculate and export outputs only for dates where required data exists.
- Save calculation runs (audit trail).

## Local run
```bash
cd streamlit_fund_rates
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
