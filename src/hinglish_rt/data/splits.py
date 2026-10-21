import hashlib
import json
from pathlib import Path

SPLITS = ["train", "validation", "test", "test_ood", "gold"]


def _u(group_id, seed):
    h = hashlib.sha1((str(seed) + ":" + str(group_id)).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def split_for_group(group_id, source, label, category, seed=42,
                    ratios=(0.70, 0.15, 0.15), ood_category=None, family=""):
    """Deterministic split for one group. Depends only on the group's id and attributes, so
    adding or removing other rows never reshuffles existing assignments."""
    if str(source).startswith("manual_"):
        return "gold"
    if ood_category and int(label) == 1 and category == ood_category:
        return "test_ood"
    u = _u(family or group_id, seed)  # synthetic families split as a whole (template-level)
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "validation"
    return "test"


def assign_splits(df, seed=42, ratios=(0.70, 0.15, 0.15), ood_category=None):
    roots = df.sort_values("id").groupby("group_id").first()
    fam = lambda r: r["family"] if isinstance(r.get("family", ""), str) else ""
    mapping = {
        gid: split_for_group(gid, r["source"], r["label"], r["category"], seed, ratios,
                             ood_category, fam(r))
        for gid, r in roots.iterrows()
    }
    df = df.copy()
    df["split"] = df["group_id"].map(mapping)
    return df, mapping


def verify_splits(df):
    """Hard consistency checks. Raises AssertionError on any leakage."""
    per_group = df.groupby("group_id")["split"].nunique()
    assert (per_group == 1).all(), "a group spans several splits"
    key = df["text"].str.lower().str.strip()
    per_text = df.assign(_k=key).groupby("_k")["split"].nunique()
    assert (per_text == 1).all(), "identical text appears in several splits"
    per_text_label = df.assign(_k=key).groupby("_k")["label"].nunique()
    assert (per_text_label == 1).all(), "identical text has conflicting labels"
    return True


def save_splits(mapping, path, seed, ratios, ood_category):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"seed": seed, "ratios": list(ratios), "ood_category": ood_category, "groups": mapping},
        indent=1), encoding="utf-8")
