import numpy as np
from sklearn.metrics import roc_auc_score

NAN = float("nan")


def summarize(y, s, thr=0.5):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else NAN
    fpr = fp / (fp + tn) if (fp + tn) else NAN
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else NAN
    auroc = float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else NAN
    return {
        "n": int(len(y)),
        "recall": recall,
        "asr": (1 - recall) if recall == recall else NAN,  # attack success rate = 1 - recall
        "fpr": fpr,
        "f1": f1,
        "auroc": auroc,
    }


def bootstrap_ci(y, s, thr=0.5, metric="recall", n_boot=1000, seed=0, alpha=0.05):
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = summarize(y[idx], s[idx], thr)[metric]
        if v == v:
            vals.append(v)
    if not vals:
        return (NAN, NAN)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
