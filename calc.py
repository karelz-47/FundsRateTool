from __future__ import annotations

from datetime import date
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

from config import (
    NAV_CURRENCY,
    SERIES_ORDER,
    PORTFOLIO_WEIGHTS,
    GUARANTEED_SERIES,
    CASH_PCT_BY_SERIES,
    DEFAULT_CASH_PCT,
)

# =============================================================================
# Helpers
# =============================================================================

def daily_yield_from_yearly(yearly: float) -> float:
    """
    Convert nominal yearly yield to an effective daily yield using compounding:
        (1 + yearly) ** (1/365) - 1
    """
    return (1.0 + float(yearly)) ** (1.0 / 365.0) - 1.0


def cash_pct_for_series(series_code: str) -> float:
    """
    Cash allocation % for a series.
    - Guaranteed series (TR_*) default to 0 unless explicitly configured.
    - Other series use CASH_PCT_BY_SERIES with fallback DEFAULT_CASH_PCT.
    """
    if series_code in CASH_PCT_BY_SERIES:
        return float(CASH_PCT_BY_SERIES[series_code])
    if series_code in GUARANTEED_SERIES:
        return 0.0
    return float(DEFAULT_CASH_PCT)


def _ensure_datetime_index(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col])
    # normalize to midnight (date-only semantics)
    return s.dt.normalize()


def _pivot_published(published_df_long: pd.DataFrame) -> pd.DataFrame:
    pub = published_df_long.copy()
    if pub.empty:
        return pd.DataFrame()
    required = {"rate_date", "series_code", "value"}
    if not required.issubset(set(pub.columns)):
        raise ValueError("published_df_long must have columns: rate_date, series_code, value")
    pub["rate_date"] = pd.to_datetime(pub["rate_date"]).dt.normalize()
    return pub.pivot_table(index="rate_date", columns="series_code", values="value", aggfunc="last")


# =============================================================================
# Core engine
# =============================================================================

def compute_outputs(
    fx_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    tr_yearly_yield: float,
    require_all_navs: bool,
    require_fx_same_day: bool,
    published_df_long: Optional[pd.DataFrame] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Computes output series for requested window.

    Key behaviours (per your latest spec):
    - Anchor date is deterministic and *dynamic*: anchor_date = (date_from - 1 day).
    - If published_df_long is provided and non-empty, the run is STRICT:
        - published history must contain ALL SERIES_ORDER series on anchor_date
        - anchor_date must be computable from NAV+FX (i.e., present in valid_dates)
      Otherwise, the function returns an EMPTY output with coverage["no_anchor"]=True.

    - Cash allocation uses *compounded* convention:
        total_return = (1 - cash_pct) * risk_return + cash_pct * cash_return
        cash_return = (1 + daily_yield) ** delta_days
      No intermediate rounding; final results are rounded to 8 decimals.

    - Missing NAV for a specific series on a day does NOT "poison" the chain.
      If the risk_return cannot be computed (missing NAV), risk_return is treated as 1.0
      for that step, so the series can resume when NAV reappears.
    """
    coverage: Dict[str, Any] = {}

    if date_from is None or date_to is None:
        raise ValueError("date_from and date_to must be provided")

    req_start = pd.Timestamp(date_from).normalize()
    req_end = pd.Timestamp(date_to).normalize()
    if req_end < req_start:
        raise ValueError("date_to must be >= date_from")

    # Determine anchor date (D-1) when published history is present
    anchor_ts: Optional[pd.Timestamp] = None
    pub_w = pd.DataFrame()
    strict_anchor = published_df_long is not None and not published_df_long.empty
    if strict_anchor:
        anchor_ts = (req_start - pd.Timedelta(days=1)).normalize()
        pub_w = _pivot_published(published_df_long)
        coverage["published_max_date"] = pub_w.index.max().date() if len(pub_w.index) else None
        coverage["anchor_date_required"] = anchor_ts.date()
    else:
        coverage["published_max_date"] = None

    # Prepare inputs
    fx = fx_df.copy()
    nav = nav_df.copy()

    if fx.empty or nav.empty:
        coverage["no_valid_dates"] = True
        return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    fx["rate_date"] = _ensure_datetime_index(fx, "rate_date")
    nav["nav_date"] = _ensure_datetime_index(nav, "nav_date")

    # FX window: need as-of lookup for all NAV dates up to req_end (and possibly anchor)
    fx = fx.sort_values("rate_date")
    fx = fx[fx["rate_date"] <= req_end].copy()

    # NAV window: must include anchor (if strict) to make it computable
    eff_start = anchor_ts if strict_anchor else req_start
    nav = nav[(nav["nav_date"] >= eff_start) & (nav["nav_date"] <= req_end)].copy()

    # Wide NAV (date x isin)
    nav_w = nav.pivot_table(index="nav_date", columns="isin", values="nav", aggfunc="last").sort_index()

    # Diagnostics lists on requested window
    nav_req = nav[(nav["nav_date"] >= req_start) & (nav["nav_date"] <= req_end)].copy()
    nav_req_w = nav_req.pivot_table(index="nav_date", columns="isin", values="nav", aggfunc="last").sort_index()

    # NAV completeness rule
    if require_all_navs:
        nav_complete_mask = ~nav_w.isna().any(axis=1)
    else:
        nav_complete_mask = ~nav_w.isna().all(axis=1)

    # FX as-of for each NAV date
    fx_asof = pd.merge_asof(
        nav_w.reset_index().rename(columns={"nav_date": "d"}).sort_values("d"),
        fx.sort_values("rate_date").rename(columns={"rate_date": "d_fx"}),
        left_on="d",
        right_on="d_fx",
        direction="backward",
    ).set_index("d")

    # FX availability rule
    if require_fx_same_day:
        fx_ok = fx_asof["d_fx"].notna() & (fx_asof["d_fx"] == fx_asof.index)
    else:
        fx_ok = fx_asof["d_fx"].notna()

    valid_dates = nav_w.index[nav_complete_mask & fx_ok]
    valid_dates = pd.DatetimeIndex(valid_dates).sort_values()

    coverage["valid_dates"] = [d.date() for d in valid_dates]

    # If strict anchoring: enforce anchor existence in published AND in valid_dates
    if strict_anchor:
        if anchor_ts not in pub_w.index:
            coverage["no_anchor"] = True
            coverage["reason"] = "published_missing_anchor_date"
            coverage["published_dates_tail"] = [d.date() for d in pub_w.index.sort_values()[-10:]]
            return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

        missing_pub = [c for c in SERIES_ORDER if (c not in pub_w.columns) or pd.isna(pub_w.at[anchor_ts, c])]
        if missing_pub:
            coverage["no_anchor"] = True
            coverage["reason"] = "published_missing_series_on_anchor"
            coverage["missing_anchor_series"] = missing_pub
            return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

        if anchor_ts not in valid_dates:
            coverage["no_anchor"] = True
            coverage["reason"] = "inputs_missing_on_anchor_date"
            # additional diag: do we have NAV? do we have FX?
            coverage["anchor_has_any_nav"] = bool(anchor_ts in nav_w.index and (~nav_w.loc[anchor_ts].isna()).any())
            coverage["anchor_has_fx"] = bool(anchor_ts in fx_asof.index and pd.notna(fx_asof.loc[anchor_ts, "d_fx"]))
            return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    # Only output within requested window
    run_dates = valid_dates[(valid_dates >= req_start) & (valid_dates <= req_end)]
    if len(run_dates) == 0:
        coverage["no_valid_dates"] = True
        return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    # Build FX factors on valid_dates
    fx_asof_valid = fx_asof.loc[valid_dates].copy()

    fx_asof_valid["huf_per_eur"] = fx_asof_valid["huf_buy"]
    fx_asof_valid["huf_per_usd"] = fx_asof_valid["huf_mid"] / fx_asof_valid["usd_sell"]

    # Build NAV in HUF on valid_dates
    nav_w_valid = nav_w.loc[valid_dates].copy()
    # Ensure all mapped ISIN columns exist (so portfolios can be computed deterministically)
    for isin in NAV_CURRENCY.keys():
        if isin not in nav_w_valid.columns:
            nav_w_valid[isin] = np.nan
    nav_w_valid = nav_w_valid.reindex(columns=sorted(nav_w_valid.columns))

    nav_huf = pd.DataFrame(index=valid_dates, dtype=float)
    for isin, ccy in NAV_CURRENCY.items():
        if ccy == "HUF":
            nav_huf[isin] = nav_w_valid[isin].astype(float)
        elif ccy == "EUR":
            nav_huf[isin] = nav_w_valid[isin].astype(float) * fx_asof_valid["huf_per_eur"].astype(float)
        elif ccy == "USD":
            nav_huf[isin] = nav_w_valid[isin].astype(float) * fx_asof_valid["huf_per_usd"].astype(float)
        else:
            raise ValueError(f"Unsupported NAV currency mapping for {isin}: {ccy}")

    # Base date for risk indices
    base_date = anchor_ts if strict_anchor else valid_dates.min()

    # Require anchor NAVs for all mapped ISINs in strict mode
    if strict_anchor:
        base_nav = nav_huf.loc[base_date]
        missing_nav = [isin for isin in NAV_CURRENCY.keys() if pd.isna(base_nav.get(isin))]
        if missing_nav:
            coverage["no_anchor"] = True
            coverage["reason"] = "missing_nav_on_anchor_date"
            coverage["missing_nav_on_anchor"] = missing_nav
            return pd.DataFrame(columns=SERIES_ORDER), {"tr_yearly_yield": tr_yearly_yield}, coverage

    # Risk indices
    base_nav_vals = nav_huf.loc[base_date]
    idx_funds = nav_huf.divide(base_nav_vals)

    dy = daily_yield_from_yearly(tr_yearly_yield)
    days_from_base = (valid_dates - pd.Timestamp(base_date)).days.astype(float)

    tr_huf_idx = pd.Series((1.0 + dy) ** days_from_base, index=valid_dates)
    base_fx_eur = float(fx_asof_valid.loc[base_date, "huf_per_eur"])
    tr_eur_idx = pd.Series((1.0 + dy) ** days_from_base, index=valid_dates) * (
        fx_asof_valid["huf_per_eur"].astype(float) / base_fx_eur
    )

    base = pd.DataFrame(index=valid_dates, dtype=float)
    # funds
    for isin in NAV_CURRENCY.keys():
        base[isin] = idx_funds[isin]
    # TR
    base["TR_HUF"] = tr_huf_idx
    base["TR_EUR"] = tr_eur_idx
    # portfolios
    for port_name, weights in PORTFOLIO_WEIGHTS.items():
        s = pd.Series(0.0, index=valid_dates, dtype=float)
        for isin, w in weights.items():
            s = s + float(w) * base[isin]
        base[port_name] = s

    # Compound cash allocation and cash yield with NaN-safe chaining
    out_idx = pd.DataFrame(index=valid_dates, columns=base.columns, dtype=float)
    out_idx.iloc[0] = base.iloc[0]  # should be 1.0 across all series at base_date

    for i in range(1, len(valid_dates)):
        t = valid_dates[i]
        t0 = valid_dates[i - 1]
        delta_days = int((t - t0).days)
        cash_ret = (1.0 + dy) ** float(delta_days)

        for col in base.columns:
            cash_pct = cash_pct_for_series(col)

            b0 = base.at[t0, col]
            b1 = base.at[t, col]

            if pd.notna(b0) and pd.notna(b1) and float(b0) != 0.0:
                risk_ret = float(b1) / float(b0)
            else:
                # Missing NAV -> keep risk part flat for that step
                risk_ret = 1.0

            total_ret = (1.0 - cash_pct) * risk_ret + cash_pct * cash_ret
            out_idx.at[t, col] = float(out_idx.at[t0, col]) * float(total_ret)

    # Convert indices to levels by anchoring to published values on anchor_date
    if strict_anchor:
        anchor_levels = pub_w.loc[base_date, SERIES_ORDER].astype(float)
        # Align columns
        out_level = out_idx[SERIES_ORDER].mul(anchor_levels, axis=1)
        coverage["anchored"] = True
        coverage["anchor_date"] = base_date.date()
    else:
        out_level = out_idx[SERIES_ORDER].copy()
        coverage["anchored"] = False
        coverage["anchor_date"] = None

    # Slice to requested output window (exclude anchor day)
    out = out_level.loc[run_dates].copy()

    # Round only at the end (8 decimals)
    out = out.round(8)

    # Additional diagnostics on requested window
    if require_all_navs:
        nav_dates_incomplete = nav_req_w.index[nav_req_w.isna().any(axis=1)]
    else:
        nav_dates_incomplete = pd.DatetimeIndex([])

    # NAV dates that have no FX (as-of) at all
    fx_asof_req = fx_asof.loc[nav_req_w.index.intersection(fx_asof.index)] if not nav_req_w.empty else pd.DataFrame()
    # nav dates missing in fx_asof entirely (no row)
    nav_dates_missing_fx = [d.date() for d in nav_req_w.index if d not in fx_asof.index or pd.isna(fx_asof.loc[d, "d_fx"])]

    coverage["nav_dates_incomplete"] = [d.date() for d in nav_dates_incomplete]
    coverage["nav_dates_missing_fx"] = nav_dates_missing_fx

    meta: Dict[str, Any] = {
        "tr_yearly_yield": float(tr_yearly_yield),
        "tr_daily_yield": float(dy),
        "require_all_navs": bool(require_all_navs),
        "require_fx_same_day": bool(require_fx_same_day),
        "base_date_for_normalization": str(pd.Timestamp(base_date).date()),
        "anchored": bool(coverage.get("anchored", False)),
        "anchor_date": str(coverage.get("anchor_date")) if coverage.get("anchor_date") else None,
        "cash_policy": {
            "default_cash_pct": DEFAULT_CASH_PCT,
            "cash_pct_by_series": CASH_PCT_BY_SERIES,
        },
    }

    return out, meta, coverage
