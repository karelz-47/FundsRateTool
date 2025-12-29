from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


_LOCALES_DIR = Path(__file__).parent / "locales"


def load_translations(lang: str) -> Dict[str, str]:
    path = _LOCALES_DIR / f"{lang}.json"
    if not path.exists():
        # fallback to EN
        path = _LOCALES_DIR / "EN.json"
    return json.loads(path.read_text(encoding="utf-8"))


def t(translations: Dict[str, str], key: str) -> str:
    return translations.get(key, f"[{key}]")
