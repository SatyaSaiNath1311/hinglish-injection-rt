import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    score: float  # P(malicious) in [0, 1]
    label: int


class Detector(ABC):
    name = "base"
    threshold = 0.5

    @abstractmethod
    def score_batch(self, texts):
        """Return a list of P(malicious) floats, one per text."""

    def detect(self, text):
        s = float(self.score_batch([text])[0])
        return Detection(s, int(s >= self.threshold))


class CachedDetector(Detector):
    """Wraps a detector with a score cache and a query counter (unique texts sent to the model)."""

    def __init__(self, inner, cache_path=None):
        self.inner = inner
        self.name = inner.name
        self.threshold = inner.threshold
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.queries = 0
        self._dirty = False
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def score_batch(self, texts):
        texts = list(texts)
        misses = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if misses:
            scores = self.inner.score_batch(misses)
            for t, s in zip(misses, scores):
                self.cache[t] = float(s)
            self.queries += len(misses)
            self._dirty = True
        return [self.cache[t] for t in texts]

    def flush(self):
        if self._dirty and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"
            )
            self._dirty = False
