import hashlib
import re
import unicodedata

import numpy as np

_WS = re.compile(r"\s+")

_EN_STOP = {
    "the", "and", "to", "of", "a", "in", "is", "you", "that", "it", "for", "with", "as",
    "are", "this", "be", "on", "your", "i", "me", "my", "what", "how", "do", "not",
    "please", "all", "from", "or", "if", "can", "will",
}
_OTHER_STOP = {
    # German
    "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "zu", "den", "mit", "auf",
    "für", "ich", "sie", "du", "wie", "was", "von", "dem", "im", "es", "sind", "bitte",
    "vergiss", "alle", "aber", "wenn",
    # French / Spanish (common function words)
    "le", "les", "et", "est", "pas", "vous", "je", "que", "el", "los", "las", "una",
    "por", "para", "con",
}


def clean_text(text):
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return _WS.sub(" ", t).strip()


def normalize_key(text):
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(text)).lower()).strip()


def stable_id(text, prefix="g"):
    return prefix + hashlib.sha1(normalize_key(text).encode("utf-8")).hexdigest()[:10]


def is_english(text):
    """Cheap filter that drops German/French/Spanish and non-Latin text."""
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127) / len(text)
    if non_ascii > 0.05:
        return False
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if not toks:
        return False
    en = sum(t in _EN_STOP for t in toks)
    other = sum(t in _OTHER_STOP for t in toks)
    return other <= en


_EXFIL = re.compile(
    r"system prompt|hidden instruction|secret|password|confidential|reveal|"
    r"print (everything|all)|repeat (your|the)|initial prompt|your instructions", re.I)
_ROLE = re.compile(
    r"you are now|pretend|act as|role-?play|\bdan\b|developer mode|jailbreak|"
    r"no restrictions|unrestricted|without (any )?restrictions", re.I)
_OVERRIDE = re.compile(
    r"ignore|disregard|forget|override|new instructions|stop following|"
    r"do not follow|don't follow|instead", re.I)
_TRIGGER = re.compile(
    r"\b(ignore|disregard|forget|override|instructions?|system prompt|previous|"
    r"jailbreak|developer mode|pretend)\b", re.I)


def categorize(text, label):
    if int(label) == 0:
        return "benign"
    if _EXFIL.search(text):
        return "exfiltration"
    if _ROLE.search(text):
        return "roleplay"
    if _OVERRIDE.search(text):
        return "override"
    return "other"


def has_trigger_words(text):
    return int(bool(_TRIGGER.search(text)))


def count_trigger_words(text):
    return len(_TRIGGER.findall(text))


def near_dedupe(texts, thr=0.95, chunk=1000):
    """Return a boolean keep-mask; drops later items whose char-ngram TF-IDF cosine to an
    earlier kept item exceeds thr."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(
        [re.sub(r"[^\w\s]", "", normalize_key(t)) for t in texts])
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sims = (X[start:end] @ X.T).toarray()
        for i in range(start, end):
            if i == 0:
                continue
            row = sims[i - start, :i]
            if (row[keep[:i]] > thr).any():
                keep[i] = False
    return keep


def near_any(candidates, references, thr=0.9, chunk=1000):
    """Boolean mask: True where a candidate has cosine similarity > thr to ANY reference text
    (punctuation-insensitive char n-gram TF-IDF). Used to keep gold prompts free of near-copies."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    candidates, references = list(candidates), list(references)
    if not candidates or not references:
        return np.zeros(len(candidates), dtype=bool)
    clean = lambda t: re.sub(r"[^\w\s]", "", normalize_key(t))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit([clean(t) for t in candidates + references])
    Xc = vec.transform([clean(t) for t in candidates])
    Xr = vec.transform([clean(t) for t in references])
    hit = np.zeros(len(candidates), dtype=bool)
    for start in range(0, len(candidates), chunk):
        sims = (Xc[start:start + chunk] @ Xr.T).toarray()
        hit[start:start + chunk] = sims.max(axis=1) > thr
    return hit
