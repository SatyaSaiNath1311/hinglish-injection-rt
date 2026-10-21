from .base import Detector


class HFClassifier(Detector):
    """Hugging Face text-classification model. Score = sum of probabilities of malicious labels."""

    def __init__(self, name, model_id, malicious_labels, threshold=0.5,
                 batch_size=32, max_length=512, token=None):
        from transformers import pipeline  # lazy import
        import torch

        self.name = name
        self.threshold = threshold
        self.malicious_labels = set(malicious_labels)
        self.batch_size = batch_size
        self.max_length = max_length
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification", model=model_id, top_k=None, device=device, token=token
        )

    def score_batch(self, texts):
        outs = self.pipe(
            list(texts), batch_size=self.batch_size, truncation=True, max_length=self.max_length
        )
        return [
            float(sum(d["score"] for d in out if d["label"] in self.malicious_labels))
            for out in outs
        ]
