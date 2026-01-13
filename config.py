from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict

# Order of series in outputs (matches your export template logic)
SERIES_ORDER = [
    "TR_HUF",
    "TR_EUR",
    "HU0000701685",
    "LU0122614208",
    "LU0605515377",
    "LU0862449690",
    "HU0000710116",
    "LU0740858492",
    "LU1088281024",
    "LU0210535034",
    "LU0329678410",
    "CONSERVATIVE",
    "BALANCED",
    "DYNAMIC",
]

# Manual NAV entry: predefined labels
FUND_LABELS: Dict[str, str] = {
    "LU0122614208": "Templeton Global Bond Fund - N (acc) USD",
    "LU0210535034": "JPMorgan Funds - Latin America Equity Fund A (acc) - USD",
    "LU0329678410": "Fidelity Funds - Emerging Asia Fund A-ACC-Euro",
    "LU0605515377": "Fidelity Funds - Global Dividend Fund A-ACC-Euro (hedged)",
    "LU0740858492": "JPMorgan Investment Funds - Global Income Fund D (acc) - EUR",
    "LU0862449690": "JPMorgan Funds - Emerging Markets Dividend Fund A (acc) - EUR",
    "LU1088281024": "Fidelity Funds - Global Multi Asset Income Fund A-ACC-HUF (hedged)",
    "HU0000701685": "TR_HUF (or underlying reference)",
    "HU0000710116": "TR_EUR (or underlying reference)",
}

# Apollon export mapping (ISIN -> Fund Code)
APOLLON_FUND_CODE_BY_ISIN = {
    "HU0000701685": "QAlphaShorttermIFund",
    "LU0122614208": "QAlphaGlobalBondIFund",
    "LU0605515377": "QAlphaGlobalDevelopedIFund",
    "LU0862449690": "QAlphaGlobalEmergingIFund",
    "HU0000710116": "QAlphaAbsoluteIFund",
    "LU0740858492": "QAlphaGlobalIncomeIFund",
    "LU1088281024": "QAlphaMixedIFund",
    "LU0210535034": "QAlphaLatinIFund",
    "LU0329678410": "QAlphaAsiaIFund",
}

# NAV currencies (as per your workbook)
NAV_CURRENCY: Dict[str, str] = {
    "HU0000701685": "HUF",
    "LU0122614208": "USD",
    "LU0605515377": "EUR",
    "LU0862449690": "EUR",
    "HU0000710116": "HUF",
    "LU0740858492": "EUR",
    "LU1088281024": "HUF",
    "LU0210535034": "USD",
    "LU0329678410": "EUR",
}

# Portfolio weights (from your corrected workbook params)
PORTFOLIO_WEIGHTS = {
    "CONSERVATIVE": {
        "HU0000701685": 0.30,
        "LU0122614208": 0.10,
        "LU0605515377": 0.05,
        "LU0862449690": 0.05,
        "HU0000710116": 0.15,
        "LU0740858492": 0.20,
        "LU1088281024": 0.15,
        "LU0210535034": 0.00,
        "LU0329678410": 0.00,
    },
    "BALANCED": {
        "HU0000701685": 0.10,
        "LU0122614208": 0.10,
        "LU0605515377": 0.15,
        "LU0862449690": 0.15,
        "HU0000710116": 0.15,
        "LU0740858492": 0.20,
        "LU1088281024": 0.15,
        "LU0210535034": 0.00,
        "LU0329678410": 0.00,
    },
    "DYNAMIC": {
        "HU0000701685": 0.00,
        "LU0122614208": 0.10,
        "LU0605515377": 0.20,
        "LU0862449690": 0.30,
        "HU0000710116": 0.15,
        "LU0740858492": 0.15,
        "LU1088281024": 0.10,
        "LU0210535034": 0.00,
        "LU0329678410": 0.00,
    },
}

# Historical normalization rate per series/ISIN (HUF value of 1 fund unit on the normalization date).
# Legacy XLS: VALUE(t) = NAV_HUF(t) / NORMALIZATION_RATE_HUF[isin]
NORMALIZATION_RATE_HUF: Dict[str, float] = {
    "HU0000701685": 4.11132000,
    "LU0122614208": 7967.76681000,
    "LU0605515377": 5044.21680000,
    "LU0862449690": 27660.75060000,
    "HU0000710116": 1.41580000,
    "LU0740858492": 39413.28720000,
    "LU1088281024": 3082.00000000,
    "LU0210535034": 6457.99590000,
    "LU0329678410": 6320.53800000,
}

# Export header rows (3-row template)
HEADER_ROW1 = [
    None, None, None,
    "NOVIS Rövid futamidejű Magyar Kötvény Eszközalap",
    "NOVIS Globális Kötvény Eszközalap",
    "NOVIS Globális Fejlett Piaci Részvény Eszközalap",
    "NOVIS Globális Fejlődő Piaci Részvény Eszközalap",
    "NOVIS Abszolút hozamú Eszközalap",
    "NOVIS Global Income Fund Eszközalap",
    "NOVIS Vegyes Eszközalap",
    "NOVIS Latin-Amerika Részvény Eszközalap",
    "NOVIS Ázsia Fejlődő Piaci Részvény Eszközalap",
    "NOVIS Mérsékelt portfólió",
    "NOVIS Kiegyensúlyozott portfólió",
    "NOVIS Dinamikus portfólió",
]

HEADER_ROW2 = [
    None, "TR_HUF", "TR_EUR",
    "HU0000701685",
    "LU0122614208",
    "LU0605515377",
    "LU0862449690",
    "HU0000710116",
    "LU0740858492",
    "LU1088281024",
    "LU0210535034",
    "LU0329678410",
    "CONSERVATIVE",
    "BALANCED",
    "DYNAMIC",
]

HEADER_ROW3 = [
    "Date", "Garant", "GarantEUR",
    "HU0000701685",
    "LU0122614208",
    "LU0605515377",
    "LU0862449690",
    "HU0000710116",
    "LU0740858492",
    "LU1088281024",
    "LU0210535034",
    "LU0329678410",
    "KONZERVATIV",
    "BALANCED",
    "DYNAMIC",
]

# TR parameters (as in your workbook)
TR_YEARLY_YIELD_DEFAULT = 0.04
TR_HUF_BASE_DATE = date(2014, 10, 1)
TR_EUR_BASE_DATE = date(2015, 6, 1)
TR_EUR_BASE_FX_FALLBACK = 313.03  # used if FX missing for 2015-06-01

ROUND_DECIMALS = 5

# config.py

# 1) Guaranteed fund series codes (FORCE cash_pct=0.0)
GUARANTEED_SERIES_CODES = {
    # Replace these keys with your actual SERIES_ORDER codes for guaranteed funds
    "NOVIS_GAR_HUF",
    "NOVIS_GAR_EUR",
}

GUARANTEED_SERIES = GUARANTEED_SERIES_CODES

# 2) Cash allocation per series (non-guaranteed)
# Values are constants and easy to edit.
# Any series not listed here will fall back to DEFAULT_CASH_PCT.
DEFAULT_CASH_PCT = 0.00

CASH_PCT_BY_SERIES = {
    # Set to 0.05 (5%) for the non-guaranteed series that should use cash damping.
    # Replace keys with YOUR actual series codes from SERIES_ORDER.
    "LU0329678410": 0.05,       # LU0329678410 FID_EM_ASIA
    "LU0605515377": 0.05,  # LU0605515377 FID_GLOB_DIV_HDG
    "LU0740858492": 0.05,   # LU0740858492 JPM_GLOB_INCOME
    "LU0862449690": 0.05,        # LU0862449690 JPM_EM_DIV
    "LU1088281024": 0.05,   # LU1088281024 FID_MULTI_ASSET

    # Everything else (including Templeton, HOLD/HU funds, JPM LatAm) effectively 0.00 by default,
    # and guaranteed funds are hard-forced to 0.00 via GUARANTEED_SERIES_CODES.
}






