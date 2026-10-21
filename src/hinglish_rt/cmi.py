import re

from .lexicon import HI_ROMAN_WORDS

_DEV = re.compile(r"[\u0900-\u097F]")
_TOKEN = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def tag_token(tok):
    if _DEV.search(tok):
        return "hi"
    t = tok.lower()
    if t.isdigit():
        return "univ"
    if t in HI_ROMAN_WORDS:
        return "hi"
    return "en"


def cmi(text):
    """Code-Mixing Index (Das and Gamback style): 0 = monolingual, higher = more mixed."""
    tags = [tag_token(t) for t in _TOKEN.findall(text)]
    n = len(tags)
    u = tags.count("univ")
    if n - u <= 0:
        return 0.0
    dominant = max(tags.count("hi"), tags.count("en"))
    return 100.0 * (1.0 - dominant / (n - u))


def script_of(text):
    return "devanagari" if _DEV.search(text) else "roman"
