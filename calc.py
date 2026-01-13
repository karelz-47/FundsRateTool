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
    NORMALIZATION_RATE_HUF,
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
    use_watermark_anchor: bool = True,  # NEW
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

    # --- Effective start: include an anchor day BEFORE the requested start (for historical normalization) ---
    # We compute outputs for the requested window, but keep the historical level of each series.
    # Anchor selection:
    #   - cutoff = (requested_start - 1 day)
    #   - choose the latest published date <= cutoff where ALL SERIES_ORDER have published values
    #   - extend input window to include that anchor date (so NAV/FX returns from anchor -> start are computable)
    anchor_required = None
    if req_start is not None:
        anchor_required = (req_start - pd.Timedelta(days=1)).normalize()

    anchor_ts_selected = None
    if published_df_long is not None and not published_df_long.empty and anchor_required is not None:
        pub_tmp = published_df_long.copy()
        pub_tmp["rate_date"] = pd.to_datetime(pub_tmp["rate_date"]).dt.normalize()
        pub_w0 = pub_tmp.pivot_table(
            index="rate_date", columns="series_code", values="value", aggfunc="last"
        ).sort_index()

        candidates = pub_w0.index[pub_w0.index <= anchor_required]
        for cand in reversed(candidates):
            missing = [c for c in SERIES_ORDER if (c not in pub_w0.columns) or pd.isna(pub_w0.at[cand, c])]
            if not missing:
                anchor_ts_selected = cand
                break

    eff_start = (
        anchor_ts_selected
        if anchor_ts_selected is not None
        else (anchor_required if anchor_required is not None else req_start)
    )

    nav_eff = nav.copy()
    if eff_start is not None:
        nav_eff = nav_eff[nav_eff["nav_date"] >= eff_start]
    if req_end is not None:
        nav_eff = nav_eff[nav_eff["nav_date"] <= req_end]

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
    nav_w_valid = nav_w.loc[valid_dates].sort_index().copy()

    # OLD_XLS TAB2 parity: NAV(t) = RAW_NAV(t) if exists else NAV(t-1)
    nav_w_valid = nav_w_valid.ffill()

    fx_asof_valid = fx_asof.set_index("date").loc[valid_dates].copy()

    # Diagnostics: where RAW_NAV was missing but NAV was carried forward
    raw_missing = nav_w.loc[valid_dates].isna()
    after_missing = nav_w_valid.isna()
    carried = raw_missing & ~after_missing
    coverage["nav_carried_forward_counts"] = carried.sum().to_dict()
    coverage["nav_carried_forward_days"] = int(carried.any(axis=1).sum())

    # FX factors per date (Excel parity)
    fx_asof_valid["huf_per_eur"] = fx_asof_valid["huf_buy"]
    fx_asof_valid["huf_per_usd"] = fx_asof_valid["huf_mid"] / fx_asof_valid["usd_sell"]

    # TR series on valid dates
    dy = daily_yield_from_yearly(tr_yearly_yield)

    # 1) Both TR_HUF and TR_EUR use the SAME inception for day counting
    day_diff = (valid_dates.date - TR_HUF_BASE_DATE)
    days = np.array([d.days for d in day_diff], dtype=float)

    tr_huf = pd.Series((1.0 + dy) ** days, index=valid_dates)

    # 2) TR_EUR = TR_HUF * FX(t) / FX(base)
    # Use HUF MID for TR_EUR parity (deviza_stred_huf)
    huf_per_eur_mid = fx_asof_valid["huf_mid"]

    base_fx = float(TR_EUR_BASE_FX_FALLBACK)
    if TR_EUR_BASE_DATE in set(valid_dates.date.tolist()):
       base_fx = float(huf_per_eur_mid.loc[pd.Timestamp(TR_EUR_BASE_DATE)])

    tr_eur = tr_huf * (huf_per_eur_mid / base_fx)

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

    # Normalize each fund using a fixed historical normalization rate (HUF per unit).
    # Legacy XLS: VALUE(t) = NAV_HUF(t) / NORMALIZATION_RATE_HUF[isin].
    base_date = valid_dates.min()
    idx_funds = pd.DataFrame(index=valid_dates, columns=nav_huf.columns, dtype=float)

    for isin in nav_huf.columns:
        norm_rate = NORMALIZATION_RATE_HUF.get(isin)

        if norm_rate is None:
            # Fallback: keeps the app running, but WILL NOT match OLD_XLS unless you provide norm rates.
            first_valid = nav_huf[isin].first_valid_index()
            if first_valid is None:
                idx_funds[isin] = np.nan
                continue
            norm_rate = float(nav_huf.at[first_valid, isin])
            coverage.setdefault("normalization_fallbacks", {})[isin] = {
                "used_date": str(pd.Timestamp(first_valid).date()),
                "used_norm_rate_huf": norm_rate,
            }

        idx_funds[isin] = nav_huf[isin] / float(norm_rate)

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

    # Fund Rate (FR) convention (legacy XLS parity):
    #   FR(t) = FR(t-1) + (VALUE(t) - VALUE(t-1)) * (1 - cash_pct) + cash_accrual
    # Anchoring is done by setting FR(anchor) = published/backfilled FR(anchor), then chaining forward.
    out = pd.DataFrame(index=valid_dates, columns=base.columns, dtype=float)

    if published_df_long is not None and not published_df_long.empty:
        pub = published_df_long.copy()
        required_cols = {"rate_date", "series_code", "value"}
        if not required_cols.issubset(set(pub.columns)):
            raise ValueError("published_df_long must have columns: rate_date, series_code, value")

        pub["rate_date"] = pd.to_datetime(pub["rate_date"]).dt.normalize()
        pub_w = pub.pivot_table(index="rate_date", columns="series_code", values="value", aggfunc="last")

        coverage["published_max_date"] = pub_w.index.max().date() if len(pub_w.index) else None

        if anchor_required is None:
            raise ValueError("date_from is required for strict anchoring")

        anchor_required_ts = anchor_required
        coverage["anchor_required_date"] = anchor_required_ts.date()

        # Anchor per series: latest date <= anchor_required where BOTH published FR and VALUE exist
        common_dates = sorted(set(base.index).intersection(set(pub_w.index)))
        common_dates = [d for d in common_dates if d <= anchor_required_ts]

        anchor_by_series = {}
        for col in base.columns:
            for cand in reversed(common_dates):
                pub_val = pub_w.at[cand, col] if (col in pub_w.columns and cand in pub_w.index) else np.nan
                val = base.at[cand, col] if (cand in base.index and col in base.columns) else np.nan
                if pd.isna(pub_val) or pd.isna(val):
                    continue
                out.at[cand, col] = float(pub_val)
                anchor_by_series[col] = cand
                break

        missing_anchor = [c for c in base.columns if c not in anchor_by_series]
        if missing_anchor:
            coverage["no_anchor"] = True
            coverage["missing_anchor_series"] = missing_anchor
            meta = {
                "tr_yearly_yield": tr_yearly_yield,
                "require_all_navs": require_all_navs,
                "require_fx_same_day": require_fx_same_day,
                "base_date_for_normalization": str(base_date.date()),
            }
            return pd.DataFrame(columns=SERIES_ORDER), meta, coverage

        # Chain forward using VALUE deltas
        for col, a in anchor_by_series.items():
            cash_pct = cash_pct_for_series(col)
            ai = base.index.get_loc(a)
            for i in range(ai + 1, len(valid_dates)):
                t = valid_dates[i]
                t0 = valid_dates[i - 1]

                prev_fr = out.at[t0, col]
                if pd.isna(prev_fr):
                    continue

                v0 = base.at[t0, col]
                v = base.at[t, col]
                if pd.isna(v0) or pd.isna(v):
                    out.at[t, col] = np.nan
                    continue

                delta_days = (t.date() - t0.date()).days
                if delta_days <= 0:
                    delta_days = 1
                cash_growth = (1.0 + dy) ** float(delta_days)

                fr = float(prev_fr) + (float(v) - float(v0)) * (1.0 - cash_pct)

                # Interest on the cash portion ONLY for TR series (TR_HUF, TR_EUR)
                if cash_pct > 0.0 and col in ("TR_HUF", "TR_EUR"):
                    cash_growth = (1.0 + dy) ** float(delta_days)
                    fr += float(prev_fr) * cash_pct * (cash_growth - 1.0)

                out.at[t, col] = fr

        coverage["anchored"] = True
        coverage["anchor_by_series"] = {k: v.date().isoformat() for k, v in anchor_by_series.items()}

    else:
        # No published history: fall back to VALUE directly (still slice to requested window later)
        out = base.copy()
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



