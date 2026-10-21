import difflib
import math


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


def score_logit(s, eps=1e-7):
    """Log-odds of a detector probability. Detectors like ProtectAI saturate near 0/1 (1 - p is
    often ~1e-7), so the learning signal lives in log-odds, not in the raw probability."""
    s = min(max(float(s), eps), 1.0 - eps)
    return math.log(s / (1.0 - s))


def compute_reward(prev_logit, logit, intent, changed, success, alpha, beta, gamma, tau,
                   success_bonus, scale=8.0):
    """Progress-based reward.

    + alpha * (drop in detector log-odds) / scale   (dense signal toward evasion)
    - gamma per step, and one more if the edit changed nothing (wasted step, no new info)
    - beta * max(0, tau - intent)                  (penalty only when meaning drifts too far)
    + success_bonus when evaded with intent >= tau

    There is deliberately NO positive per-step term: a positive per-step reward makes ending the
    episode (by succeeding) unattractive, and PPO then learns to avoid success.
    """
    r = alpha * (prev_logit - logit) / scale - gamma
    if not changed:
        r -= gamma
    r -= beta * max(0.0, tau - intent)
    if success:
        r += success_bonus
    return r
