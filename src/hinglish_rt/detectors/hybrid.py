from .base import Detector


class HybridDetector(Detector):
    """Model + regex hybrid (like PromptGuard): score = max(model score, regex score)."""

    def __init__(self, name, model_detector, regex_detector, threshold=0.5):
        self.name = name
        self.threshold = threshold
        self.model = model_detector
        self.regex = regex_detector

    def score_batch(self, texts):
        texts = list(texts)
        m = self.model.score_batch(texts)
        r = self.regex.score_batch(texts)
        return [max(a, b) for a, b in zip(m, r)]
