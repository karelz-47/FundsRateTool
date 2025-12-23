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
