import re


def normalize_whitespace(s: str) -> str:
    """Collapse whitespace runs into single spaces and strip."""
    return " ".join((s or "").split()).strip()


def normalize_for_comparison(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy matching."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
