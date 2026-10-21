import difflib


class LexicalIntentJudge:
    """Cheap string-similarity stand-in. Smoke tests only, NOT valid for real experiments."""

    name = "lexical"

    def __call__(self, original, rewritten):
        return difflib.SequenceMatcher(None, original.lower(), rewritten.lower()).ratio()


class EmbeddingIntentJudge:
    """Multilingual sentence-embedding cosine similarity (a proxy for meaning preservation).

    For publishable results also add an independent malicious-intent check (classifier or LLM
    judge) so an evasion only counts if the rewrite is still an injection.
    """

    name = "embedding"

    def __init__(self, model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device=None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_id, device=device)

    def __call__(self, original, rewritten):
        e = self.model.encode([original, rewritten], normalize_embeddings=True)
        return float(max(0.0, min(1.0, float(e[0] @ e[1]))))


def compute_reward(score, intent, cost, success, alpha, beta, gamma, success_bonus):
    r = alpha * (1.0 - score) + beta * intent - gamma * cost
    if success:
        r += success_bonus
    return r
