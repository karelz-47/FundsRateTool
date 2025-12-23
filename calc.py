from __future__ import annotations

from datetime import date
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from config import (
    NAV_CURRENCY, PORTFOLIO_WEIGHTS,
    TR_YEARLY_YIELD_DEFAULT, TR_HUF_BASE_DATE, TR_EUR_BASE_DATE, TR_EUR_BASE_FX_FALLBACK,
    ROUND_DECIMALS, SERIES_ORDER
)


def daily_yield_from_yearly(yearly: float) -> float:
    return (1.0 + yearly) ** (1.0 / 365.0) - 1.0


def compute_outputs(
    fx_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    cash_df: Optional[pd.DataFrame],
    tr_yearly_yield: float = TR_YEARLY_YIELD_DEFAULT,
    require_all_navs: bool = True,
    require_fx_same_day: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    """
    Outputs are only for valid dates where:
      - NAV exists for the date (and is complete if require_all_navs)
      - FX exists for the date (same-day if require_fx_same_day else as-of last FX <= NAV date)

    Returns:
      out_df: Date-indexed DataFrame with columns in SERIES_ORDER (only valid dates)
      meta: parameters used
      coverage: diagnostics lists
    """

    # Allow independent import (no hard failure)
    if nav_df is None or nav_df.empty:
        coverage = {
            "valid_dates": [],
            "nav_dates_missing_fx": [],
            "nav_dates_incomplete": [],
            "nav_dates_in_db": [],
            "fx_dates_in_db": sorted(pd.to_datetime(fx_df["rate_date"]).dt.date.unique().tolist()) if fx_df is not None and not fx_df.empty else [],
            "require_all_navs": require_all_navs,
            "require_fx_same_day": require_fx_same_day,
        }
        return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    if fx_df is None or fx_df.empty:
        nav_dates = sorted(pd.to_datetime(nav_df["nav_date"]).dt.date.unique().tolist())
        coverage = {
            "valid_dates": [],
            "nav_dates_missing_fx": nav_dates,
            "nav_dates_incomplete": [],
            "nav_dates_in_db": nav_dates,
            "fx_dates_in_db": [],
            "require_all_navs": require_all_navs,
            "require_fx_same_day": require_fx_same_day,
        }
        return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    fx = fx_df.copy()
    fx["rate_date"] = pd.to_datetime(fx["rate_date"]).dt.normalize()
    fx = fx.sort_values("rate_date")

    nav = nav_df.copy()
    nav["nav_date"] = pd.to_datetime(nav["nav_date"]).dt.normalize()
    nav = nav.sort_values(["nav_date", "isin"])

    required_isins = list(NAV_CURRENCY.keys())

    # NAV wide table
    nav_w = nav.pivot_table(index="nav_date", columns="isin", values="nav", aggfunc="last").sort_index()
    for isin in required_isins:
        if isin not in nav_w.columns:
            nav_w[isin] = np.nan
    nav_w = nav_w[required_isins]

    # NAV completeness by date
    if require_all_navs:
        nav_complete_mask = ~nav_w.isna().any(axis=1)
    else:
        nav_complete_mask = ~nav_w.isna().all(axis=1)

    nav_dates_incomplete = nav_w.index[~nav_complete_mask].date.tolist()

    # FX as-of map for NAV dates
    dates = pd.DataFrame({"date": nav_w.index}).sort_values("date")

    fx_for_asof = fx.rename(columns={"rate_date": "date"}).sort_values("date")
    fx_asof = pd.merge_asof(dates, fx_for_asof, on="date", direction="backward")

    fx_required_cols = ["huf_buy", "huf_mid", "usd_sell"]
    fx_ok = ~fx_asof[fx_required_cols].isna().any(axis=1)

    if require_fx_same_day:
        fx_days = set(fx["rate_date"].dt.date.tolist())
        fx_ok = fx_ok & fx_asof["date"].dt.date.isin(fx_days)

    nav_dates_missing_fx = fx_asof.loc[~fx_ok, "date"].dt.date.tolist()

    # Valid dates are those with NAV completeness + FX availability
    valid_mask = fx_ok.values & nav_complete_mask.reindex(nav_w.index).values
    valid_dates = nav_w.index[valid_mask]

    coverage = {
        "valid_dates": valid_dates.date.tolist(),
        "nav_dates_missing_fx": nav_dates_missing_fx,
        "nav_dates_incomplete": nav_dates_incomplete,
        "nav_dates_in_db": nav_w.index.date.tolist(),
        "fx_dates_in_db": fx["rate_date"].dt.date.unique().tolist(),
        "require_all_navs": require_all_navs,
        "require_fx_same_day": require_fx_same_day,
    }

    if len(valid_dates) == 0:
        return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    # Filter to valid dates only
    nav_w_valid = nav_w.loc[valid_dates].copy()
    fx_asof_valid = fx_asof.set_index("date").loc[valid_dates].copy()

    # FX factors per date (Excel parity)
    fx_asof_valid["huf_per_eur"] = fx_asof_valid["huf_buy"]
    fx_asof_valid["huf_per_usd"] = fx_asof_valid["huf_mid"] / fx_asof_valid["usd_sell"]

    # TR series on valid dates
    dy = daily_yield_from_yearly(tr_yearly_yield)

    day_diff_huf = (valid_dates.date - TR_HUF_BASE_DATE)
    days_huf = np.array([d.days for d in day_diff_huf], dtype=float)
    tr_huf = pd.Series((1.0 + dy) ** days_huf, index=valid_dates)

    day_diff_eur = (valid_dates.date - TR_EUR_BASE_DATE)
    days_eur = np.array([d.days for d in day_diff_eur], dtype=float)
    tr_eur_raw = (1.0 + dy) ** days_eur

    # base FX for TR_EUR normalization
    base_fx = TR_EUR_BASE_FX_FALLBACK
    if TR_EUR_BASE_DATE in set(valid_dates.date.tolist()):
        base_fx = float(fx_asof_valid.loc[pd.Timestamp(TR_EUR_BASE_DATE), "huf_per_eur"])
    else:
        # If base date not in valid range, try to get as-of FX for that base date from full fx table
        # (still deterministic; optional enhancement)
        pass

    tr_eur = pd.Series(tr_eur_raw * fx_asof_valid["huf_per_eur"].to_numpy() / base_fx, index=valid_dates)

    # Convert NAV to HUF
    nav_huf = pd.DataFrame(index=valid_dates, columns=required_isins, dtype=float)
    for isin, ccy in NAV_CURRENCY.items():
        if ccy == "HUF":
            nav_huf[isin] = nav_w_valid[isin]
        elif ccy == "EUR":
            nav_huf[isin] = nav_w_valid[isin] * fx_asof_valid["huf_per_eur"]
        elif ccy == "USD":
            nav_huf[isin] = nav_w_valid[isin] * fx_asof_valid["huf_per_usd"]
        else:
            raise ValueError(f"Unsupported currency for {isin}: {ccy}")

    # Normalize each fund to 1.0 at the first valid date (with complete NAV & FX)
    base_date = valid_dates.min()
    base_vals = nav_huf.loc[base_date]
    idx_funds = nav_huf.divide(base_vals)

    # Portfolios (return-chaining across valid dates)
    def compute_portfolio(weights: Dict[str, float]) -> pd.Series:
        out = pd.Series(index=valid_dates, dtype=float)
        out.iloc[0] = 1.0
        for i in range(1, len(valid_dates)):
            t = valid_dates[i]
            t0 = valid_dates[i - 1]
            ratios = []
            wts = []
            for isin, w in weights.items():
                if w == 0:
                    continue
                a = float(idx_funds.at[t0, isin])
                b = float(idx_funds.at[t, isin])
                ratios.append(b / a)
                wts.append(w)
            out.iloc[i] = out.iloc[i - 1] * float(np.dot(wts, ratios))
        return out

    port_cons = compute_portfolio(PORTFOLIO_WEIGHTS["CONSERVATIVE"])
    port_bal = compute_portfolio(PORTFOLIO_WEIGHTS["BALANCED"])
    port_dyn = compute_portfolio(PORTFOLIO_WEIGHTS["DYNAMIC"])

    base = pd.DataFrame(index=valid_dates)
    base["TR_HUF"] = tr_huf
    base["TR_EUR"] = tr_eur
    for isin in required_isins:
        base[isin] = idx_funds[isin]
    base["CONSERVATIVE"] = port_cons
    base["BALANCED"] = port_bal
    base["DYNAMIC"] = port_dyn

    # Cash allocation delta-damping (optional)
    if cash_df is None or cash_df.empty:
        cash_map = None
    else:
        cash = cash_df.copy()
        cash["alloc_date"] = pd.to_datetime(cash["alloc_date"]).dt.normalize()
        cash_map = (
            cash.sort_values("alloc_date")
                .pivot_table(index="alloc_date", columns="series_code", values="cash_pct", aggfunc="last")
        )

    out = pd.DataFrame(index=valid_dates, columns=base.columns, dtype=float)
    out.iloc[0] = base.iloc[0]

    for i in range(1, len(valid_dates)):
        t = valid_dates[i]
        t0 = valid_dates[i - 1]
        for col in base.columns:
            cash_pct = 0.0
            if cash_map is not None and col in cash_map.columns and t in cash_map.index:
                v = cash_map.at[t, col]
                if not pd.isna(v):
                    cash_pct = float(v)
            out.at[t, col] = out.at[t0, col] + (base.at[t, col] - base.at[t0, col]) * (1.0 - cash_pct)

    out = out.round(ROUND_DECIMALS)
    out = out[SERIES_ORDER]

    meta = {
        "tr_yearly_yield": tr_yearly_yield,
        "tr_daily_yield": dy,
        "tr_huf_base_date": str(TR_HUF_BASE_DATE),
        "tr_eur_base_date": str(TR_EUR_BASE_DATE),
        "tr_eur_base_fx_used": base_fx,
        "base_date_for_normalization": str(base_date.date()),
        "require_all_navs": require_all_navs,
        "require_fx_same_day": require_fx_same_day,
    }

    return out, meta, coverage
