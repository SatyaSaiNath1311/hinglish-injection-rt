import os
from pathlib import Path

from .base import CachedDetector
from .hf_classifier import HFClassifier
from .hybrid import HybridDetector
from .regex_detector import RegexDetector


def _build_one(spec, token):
    kind = spec["type"]
    thr = spec.get("threshold", 0.5)
    if kind == "regex":
        return RegexDetector(name=spec["name"], threshold=thr)
    if kind == "hf":
        return HFClassifier(
            name=spec["name"], model_id=spec["model_id"],
            malicious_labels=spec["malicious_labels"], threshold=thr, token=token,
        )
    if kind == "hybrid":
        model = HFClassifier(
            name=spec["name"] + "_model", model_id=spec["model_id"],
            malicious_labels=spec["malicious_labels"], threshold=thr, token=token,
        )
        return HybridDetector(spec["name"], model, RegexDetector(name=spec["name"] + "_regex"), thr)
    raise ValueError("Unknown detector type: " + str(kind))


def build_detectors(cfg, only=None, cache_dir=None):
    """Return {name: CachedDetector}. Disabled detectors load only if named in `only`."""
    token = os.environ.get("HF_TOKEN")
    out = {}
    for spec in cfg["detectors"]:
        named = bool(only) and spec["name"] in only
        if only and not named:
            continue
        if not spec.get("enabled", True) and not named:
            continue
        det = _build_one(spec, token)
        cache_path = Path(cache_dir) / (spec["name"] + ".json") if cache_dir else None
        out[spec["name"]] = CachedDetector(det, cache_path)
    return out
