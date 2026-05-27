import gettext
import os
from pathlib import Path
from typing import Optional

LOCALES_DIR = Path(__file__).resolve().parent / "locales"


def normalize_lang(lang_value: Optional[str]) -> str:
    """Normalizes language values like 'en_US.UTF-8' to 'en'."""
    if not lang_value:
        return "es"

    return lang_value.split(".")[0].split("_")[0].strip() or "es"


def configure_language(lang: Optional[str]) -> gettext.NullTranslations:
    """Configures process-wide i18n and returns the active translation."""
    normalized_lang = normalize_lang(lang)
    os.environ["APP_LANG"] = normalized_lang
    os.environ["LANGUAGE"] = normalized_lang
    translation = gettext.translation(
        "messages",
        localedir=LOCALES_DIR,
        languages=[normalized_lang],
        fallback=True,
    )
    translation.install()
    return translation


def get_translation() -> gettext.NullTranslations:
    """Builds a translation object using the current environment language."""
    lang = os.environ.get("APP_LANG") or os.environ.get("LANG")
    return gettext.translation(
        "messages",
        localedir=LOCALES_DIR,
        languages=[normalize_lang(lang)],
        fallback=True,
    )


def get_translator():
    """Returns a gettext-compatible translator callable."""
    return get_translation().gettext
