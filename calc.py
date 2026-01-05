from __future__ import annotations

from datetime import date
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from config import (
    NAV_CURRENCY,
    PORTFOLIO_WEIGHTS,
    TR_YEARLY_YIELD_DEFAULT,
    TR_HUF_BASE_DATE,
    TR_EUR_BASE_DATE,
    TR_EUR_BASE_FX_FALLBACK,
    ROUND_DECIMALS,
    SERIES_ORDER,
    GUARANTEED_SERIES,
    DEFAULT_CASH_PCT,
    CASH_PCT_BY_SERIES,
)


def daily_yield_from_yearly(yearly: float) -> float:
    return (1.0 + yearly) ** (1.0 / 365.0) - 1.0


def cash_pct_for_series(series_code: str) -> float:
    # Guaranteed funds => 0% cash allocation per your policy
    if series_code in GUARANTEED_SERIES:
        return 0.0
    return float(CASH_PCT_BY_SERIES.get(series_code, DEFAULT_CASH_PCT))


def compute_outputs(
    fx_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    tr_yearly_yield: float = TR_YEARLY_YIELD_DEFAULT,
    require_all_navs: bool = True,
    require_fx_same_day: bool = False,
    published_df_long: Optional[pd.DataFrame] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    """
    Outputs are only for valid dates where:
      - NAV exists for the date (and is complete if require_all_navs)
      - FX exists for the date (same-day if require_fx_same_day else as-of last FX <= NAV date)

    STRICT ANCHORING:
      If published_df_long is provided (and non-empty), outputs are "anchor-scaled" so that
      the computed value on the latest anchor date equals the published value on that date.
      Anchor date must be:
        - a date that exists in computed valid_dates
        - AND a date that exists in published history
        - AND on which ALL output series have a published value
      If no such date exists: return EMPTY output (strict behavior).
    """

    fx = fx_df.copy()
    nav = nav_df.copy()

    # Normalize date columns
    fx["rate_date"] = pd.to_datetime(fx["rate_date"]).dt.normalize()
    nav["nav_date"] = pd.to_datetime(nav["nav_date"]).dt.normalize()

    # --- Requested window (for diagnostics and final slicing) ---
    req_start = pd.Timestamp(date_from) if date_from else None
    req_end = pd.Timestamp(date_to) if date_to else None

    nav_req = nav.copy()
    if req_start is not None:
        nav_req = nav_req[nav_req["nav_date"] >= req_start]
    if req_end is not None:
        nav_req = nav_req[nav_req["nav_date"] <= req_end]

    # Keep FX up to req_end (no need to keep future FX). Do NOT cut FX lower-bound because as-of merge
    # may legitimately use a prior FX for the first requested NAV date.
    fx_req = fx.copy()
    if req_end is not None:
        fx_req = fx_req[fx_req["rate_date"] <= req_end]

    # --- Optional effective start extension (only when anchoring is used) ---
    # If the user requests a window strictly AFTER the published watermark, we must include the watermark
    # date in the computation so scaling can be applied.
    eff_start = req_start
    watermark_ts = None

    if published_df_long is not None and not published_df_long.empty:
        pub_tmp = published_df_long.copy()
        pub_tmp["rate_date"] = pd.to_datetime(pub_tmp["rate_date"]).dt.normalize()
        watermark_ts = pub_tmp["rate_date"].max()

        # Only extend if the requested start exists and is after watermark
        if eff_start is not None and watermark_ts is not None and eff_start > watermark_ts:
            eff_start = watermark_ts

    nav_eff = nav.copy()
    if eff_start is not None:
        nav_eff = nav_eff[nav_eff["nav_date"] >= eff_start]
    if req_end is not None:
        nav_eff = nav_eff[nav_eff["nav_date"] <= req_end]

    # From here onward use nav_eff / fx_req for computation; use nav_req for diagnostics lists.
    nav = nav_eff
    fx = fx_req

    # Diagnostics NAV wide (requested window)
    nav_w_req = nav_req.pivot_table(index="nav_date", columns="isin", values="nav", aggfunc="last").sort_index()

    # Computation NAV wide (may start earlier than requested due to watermark anchoring)
    nav_w = nav.pivot_table(index="nav_date", columns="isin", values="nav", aggfunc="last").sort_index()

    # Diagnostics: NAV completeness by date
    if require_all_navs:
        nav_complete_mask_req = ~nav_w_req.isna().any(axis=1)
        nav_complete_mask = ~nav_w.isna().any(axis=1)
    else:
        nav_complete_mask_req = ~nav_w_req.isna().all(axis=1)
        nav_complete_mask = ~nav_w.isna().all(axis=1)

    nav_dates_incomplete = nav_w_req.index[~nav_complete_mask_req].date.tolist()

    # FX as-of join (Excel parity): last FX <= NAV date
    fx_sorted = fx.sort_values("rate_date").copy()
        # Requested-window FX coverage diagnostics
    nav_dates_req = pd.DataFrame({"date": nav_w_req.index})
    fx_asof_req = pd.merge_asof(
        nav_dates_req.sort_values("date"),
        fx_sorted.sort_values("rate_date"),
        left_on="date",
        right_on="rate_date",
        direction="backward",
    )
    fx_ok_req = ~fx_asof_req["rate_date"].isna()
    if require_fx_same_day:
        fx_days = set(fx["rate_date"].dt.date.tolist())
        fx_ok_req = fx_ok_req & fx_asof_req["date"].dt.date.isin(fx_days)
    nav_dates_missing_fx = fx_asof_req.loc[~fx_ok_req, "date"].dt.date.tolist()

    # Computation-window FX as-of (used for actual conversions)
    nav_dates = pd.DataFrame({"date": nav_w.index})
    fx_asof = pd.merge_asof(
        nav_dates.sort_values("date"),
        fx_sorted.sort_values("rate_date"),
        left_on="date",
        right_on="rate_date",
        direction="backward",
    )
    fx_ok = ~fx_asof["rate_date"].isna()
    if require_fx_same_day:
        fx_days = set(fx["rate_date"].dt.date.tolist())
        fx_ok = fx_ok & fx_asof["date"].dt.date.isin(fx_days)

    # Valid dates are those with NAV completeness + FX availability
    valid_mask = fx_ok.values & nav_complete_mask.reindex(nav_w.index).values
    valid_dates = nav_w.index[valid_mask]

    coverage = {
        "valid_dates": valid_dates.date.tolist(),
        "nav_dates_missing_fx": nav_dates_missing_fx,
        "nav_dates_incomplete": nav_dates_incomplete,
        "nav_dates_in_db": nav_w_req.index.date.tolist(),
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

    tr_eur = pd.Series(tr_eur_raw, index=valid_dates) * (fx_asof_valid["huf_per_eur"] / float(base_fx))

    # Convert fund NAVs to HUF
    nav_huf = pd.DataFrame(index=valid_dates, columns=nav_w_valid.columns, dtype=float)
    for isin in nav_w_valid.columns:
        ccy = NAV_CURRENCY.get(isin)
        if ccy is None:
            raise ValueError(f"ISIN {isin} not found in NAV_CURRENCY mapping")
        ccy = str(ccy).upper()

        if ccy == "HUF":
            nav_huf[isin] = nav_w_valid[isin]
        elif ccy == "EUR":
            nav_huf[isin] = nav_w_valid[isin] * fx_asof_valid["huf_per_eur"]
        elif ccy == "USD":
            nav_huf[isin] = nav_w_valid[isin] * fx_asof_valid["huf_per_usd"]
        else:
            raise ValueError(f"Unsupported currency for {isin}: {ccy}")

    # Normalize each fund to 1.0 at the first valid date (relative series)
    base_date = valid_dates.min()
    base_vals = nav_huf.loc[base_date]
    idx_funds = nav_huf.divide(base_vals)

    # Portfolios (weighted sums of fund indices)
    base = pd.DataFrame(index=valid_dates, dtype=float)

    for isin in idx_funds.columns:
        base[isin] = idx_funds[isin]

    # Add TR series
    base["TR_HUF"] = tr_huf
    base["TR_EUR"] = tr_eur

    # Portfolio composites (expected keys in PORTFOLIO_WEIGHTS)
    for port_name, weights in PORTFOLIO_WEIGHTS.items():
        s = pd.Series(0.0, index=valid_dates)
        for isin, w in weights.items():
            s = s + float(w) * idx_funds[isin]
        base[port_name] = s

    # Cash allocation delta-damping (constant policy)
    out = pd.DataFrame(index=valid_dates, columns=base.columns, dtype=float)
    out.iloc[0] = base.iloc[0]

    for i in range(1, len(valid_dates)):
        t = valid_dates[i]
        t0 = valid_dates[i - 1]
        for col in base.columns:
            cash_pct = cash_pct_for_series(col)
            out.at[t, col] = out.at[t0, col] + (base.at[t, col] - base.at[t0, col]) * (1.0 - cash_pct)

    # If published history is provided, scale (chain) outputs to the latest anchor date
    # where ALL series have published values AND the date exists in our computed valid dates.
    # Strict behavior: if no such anchor exists, return empty output (do not fabricate 1.0 series).
    if published_df_long is not None and not published_df_long.empty:
        pub = published_df_long.copy()
        required_cols = {"rate_date", "series_code", "value"}
        if not required_cols.issubset(set(pub.columns)):
            raise ValueError("published_df_long must have columns: rate_date, series_code, value")

        pub["rate_date"] = pd.to_datetime(pub["rate_date"]).dt.normalize()
        pub_w = pub.pivot_table(index="rate_date", columns="series_code", values="value", aggfunc="last")

        # Track published coverage for diagnostics
        if len(pub_w.index) > 0:
            coverage["published_max_date"] = pub_w.index.max().date()
        else:
            coverage["published_max_date"] = None

        common_dates = sorted(set(out.index).intersection(set(pub_w.index)))
        anchor_ts = None
        missing_for_best = None

        # Choose the most recent common date that has COMPLETE published values for all series
        for cand in reversed(common_dates):
            missing = [c for c in out.columns if (c not in pub_w.columns) or pd.isna(pub_w.at[cand, c])]
            if not missing:
                anchor_ts = cand
                break
            missing_for_best = missing

        if anchor_ts is None:
            coverage["no_anchor"] = True
            if common_dates:
                coverage["anchor_candidates"] = [d.date() for d in common_dates[-10:]]  # last 10 candidates
                if missing_for_best is not None:
                    coverage["missing_anchor_series"] = missing_for_best
            meta = {
                "tr_yearly_yield": tr_yearly_yield,
                "require_all_navs": require_all_navs,
                "require_fx_same_day": require_fx_same_day,
                "base_date_for_normalization": str(base_date.date()),
            }
            return pd.DataFrame(columns=SERIES_ORDER), meta, coverage

        # Scale each series so that computed value on anchor date equals published value on anchor date
        denom = out.loc[anchor_ts, out.columns]
        if (denom == 0).any():
            zeros = denom[denom == 0].index.tolist()
            raise ValueError(f"Cannot anchor-scale on {anchor_ts.date()} because computed value is 0 for: {zeros}")

        scales = pub_w.loc[anchor_ts, out.columns] / denom
        out = out.mul(scales, axis=1)
        coverage["anchored"] = True
        coverage["anchor_date"] = anchor_ts.date()
    else:
        coverage["anchored"] = False
        coverage["anchor_date"] = None
        coverage["published_max_date"] = None
        coverage["requested_date_from"] = str(date_from) if date_from else None
        coverage["requested_date_to"] = str(date_to) if date_to else None

        # Final slicing to requested window
    if req_start is not None:
        out = out[out.index >= req_start]
    if req_end is not None:
        out = out[out.index <= req_end]
    
    out = out.round(ROUND_DECIMALS)
    out = out[SERIES_ORDER]

    meta = {
        "tr_yearly_yield": tr_yearly_yield,
        "tr_daily_yield": dy,
        "tr_huf_base_date": str(TR_HUF_BASE_DATE),
        "tr_eur_base_date": str(TR_EUR_BASE_DATE),
        "tr_eur_base_fx_used": base_fx,
        "base_date_for_normalization": str(base_date.date()),
        "anchored": bool(coverage.get("anchored", False)),
        "anchor_date": str(coverage.get("anchor_date")) if coverage.get("anchor_date") else None,
        "require_all_navs": require_all_navs,
        "require_fx_same_day": require_fx_same_day,
        "cash_policy": {
            "mode": "constants_in_code",
            "guaranteed_series": sorted(list(GUARANTEED_SERIES)),
            "default_cash_pct": DEFAULT_CASH_PCT,
            "cash_pct_by_series": CASH_PCT_BY_SERIES,
        },
    }

    return out, meta, coverage

