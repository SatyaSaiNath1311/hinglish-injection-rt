import re

from .base import Detector

DEFAULT_PATTERNS = [
    r"\bignore\s+(all\s+|the\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules)",
    r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above)",
    r"\breveal\s+(your\s+|the\s+)?system\s+prompt",
    r"\bforget\s+(all|everything|your)\b",
    r"\byou\s+are\s+now\b",
]


class RegexDetector(Detector):
    """English-only keyword baseline. Score is 1.0 on any match, else 0.0."""

    def __init__(self, name="regex_baseline", patterns=None, threshold=0.5):
        self.name = name
        self.threshold = threshold
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or DEFAULT_PATTERNS)]

    def score_batch(self, texts):
        return [1.0 if any(p.search(t) for p in self.patterns) else 0.0 for t in texts]
