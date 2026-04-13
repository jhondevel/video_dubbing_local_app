from __future__ import annotations

LANGUAGE_ALIASES = {
    "es": "es",
    "español": "es",
    "spanish": "es",
    "en": "en",
    "english": "en",
    "inglés": "en",
    "fr": "fr",
    "francés": "fr",
    "french": "fr",
    "de": "de",
    "alemán": "de",
    "german": "de",
    "it": "it",
    "italiano": "it",
    "italian": "it",
    "pt": "pt",
    "portugués": "pt",
    "portuguese": "pt",
    "ru": "ru",
    "ruso": "ru",
    "russian": "ru",
    "ja": "ja",
    "japonés": "ja",
    "japanese": "ja",
    "zh": "zh",
    "chino": "zh",
    "chinese": "zh",
    "ko": "ko",
    "coreano": "ko",
    "korean": "ko",
    "ar": "ar",
    "árabe": "ar",
    "arabic": "ar",
    "hi": "hi",
    "hindi": "hi",
    "tr": "tr",
    "turco": "tr",
    "turkish": "tr",
    "pl": "pl",
    "polaco": "pl",
    "polish": "pl",
    "nl": "nl",
    "neerlandés": "nl",
    "dutch": "nl",
    "ca": "ca",
    "catalán": "ca",
    "catalan": "ca",
}

SUGGESTED_VOICES = {
    "es": "es_ES-sharvard-medium",
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-upmc-medium",
    "de": "de_DE-thorsten-medium",
    "it": "it_IT-riccardo-x_low",
    "pt": "pt_BR-edresson-medium",
    "ru": "ru_RU-ruslan-medium",
    "ja": "ja_JP-kokoro-medium",
    "zh": "zh_CN-huayan-medium",
    "ko": "ko_KR-kss-medium",
    "ar": "ar_JO-kareem-medium",
    "hi": "hi_IN-priyamvada-medium",
    "tr": "tr_TR-dfki-medium",
    "pl": "pl_PL-darkman-medium",
    "nl": "nl_NL-mls_5809-low",
    "ca": "ca_ES-upc_ona-medium",
}


def normalize_language(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[cleaned]
    raise ValueError(
        "Idioma no reconocido. Use un código como es, en, fr, de, it, pt, ru, ja, zh, ko, ar, hi, tr, pl, nl o ca."
    )
