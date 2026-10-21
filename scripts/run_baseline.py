# English-vs-Hinglish detector evaluation on the frozen master dataset.
#
#   python scripts/run_baseline.py --only regex_baseline protectai_deberta
#   python scripts/run_baseline.py --split test test_ood gold
#
# Outputs (results/tables): baseline_by_group.csv, mcnemar_pairs.csv
#          (results/raw):    baseline_per_sample.csv
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from hinglish_rt.detectors.registry import build_detectors  # noqa: E402
from hinglish_rt.eval.metrics import bootstrap_ci, summarize  # noqa: E402

GROUP_COLS = ["lang", "script", "mix_level", "cmi_bin", "category", "has_trigger_words",
              "root_trigger", "lang_x_root_trigger", "source"]


def _metrics_row(det, split_name, by, name, g, thr):
    m = summarize(g.label.values, g.score.values, thr)
    lo, hi = bootstrap_ci(g.label.values, g.score.values, thr, "recall", n_boot=200,
                          groups=g.group_id.values)
    return {"detector": det, "split": split_name, "group_by": by, "group": str(name),
            "n_pos": int((g.label == 1).sum()), "n_neg": int((g.label == 0).sum()),
            "recall": m["recall"], "asr": m["asr"], "fpr": m["fpr"], "f1": m["f1"],
            "auroc": m["auroc"], "recall_ci_lo": lo, "recall_ci_hi": hi}


def _mcnemar(det, df, thr):
    """Paired English vs Hinglish comparison per mix level (twins share group_id)."""
    rows = []
    en = df[df.lang == "en"].set_index("group_id")
    for level, hi in df[df.lang == "hinglish"].groupby("mix_level"):
        hi = hi.drop_duplicates("group_id").set_index("group_id")
        common = en.index.intersection(hi.index)
        if len(common) == 0:
            continue
        e, h = en.loc[common], hi.loc[common]
        for label, name in ((1, "malicious (recall)"), (0, "benign (false positives)")):
            mask = (e.label.values == label)
            if mask.sum() == 0:
                continue
            ce = (e.score.values[mask] >= thr)
            ch = (h.score.values[mask] >= thr)
            b = int((ce & ~ch).sum())   # flagged in English only
            c = int((~ce & ch).sum())   # flagged in Hinglish only
            p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else float("nan")
            rows.append({"detector": det, "mix_level": level, "subset": name,
                         "n_pairs": int(mask.sum()), "flagged_en_only": b,
                         "flagged_hinglish_only": c, "p_value": p})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "master.csv"))
    ap.add_argument("--split", nargs="+", default=["test", "test_ood", "gold"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = pd.read_csv(a.data)
    df = df[df.split.isin(a.split)].reset_index(drop=True)
    df["cmi_bin"] = pd.cut(df.cmi, [-0.1, 0, 20, 40, 101],
                           labels=["0", "(0,20]", "(20,40]", "(40,100]"]).astype(str)
    if "root_trigger" not in df.columns:  # older master.csv
        roots = df[df.lang == "en"].set_index("group_id")["has_trigger_words"]
        df["root_trigger"] = df["group_id"].map(roots).fillna(0).astype(int)
    df["lang_x_root_trigger"] = df.lang + "_trig" + df.root_trigger.astype(str)
    print("Evaluating on %d rows from splits %s" % (len(df), a.split))
    dets = build_detectors(cfg, only=a.only, cache_dir=ROOT / cfg["paths"]["cache_dir"])

    rows, mc, per_sample = [], [], []
    for name, det in dets.items():
        d = df.copy()
        d["score"] = det.score_batch(d.text.tolist())
        det.flush()
        d["detector"] = name
        per_sample.append(d[["detector", "id", "group_id", "split", "label", "lang", "score"]])
        for split_name, gs in [("all", d)] + list(d.groupby("split")):
            rows.append(_metrics_row(name, split_name, "overall", "all", gs, det.threshold))
            for col in GROUP_COLS:
                for val, g in gs.groupby(col):
                    rows.append(_metrics_row(name, split_name, col, val, g, det.threshold))
        mc.extend(_mcnemar(name, d, det.threshold))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "baseline_by_group.csv", index=False)
    pd.DataFrame(mc).to_csv(out / "mcnemar_pairs.csv", index=False)
    pd.concat(per_sample).to_csv(raw / "baseline_per_sample.csv", index=False)

    show = res[(res.split == "all") & (res.group_by == "lang")]
    print(show[["detector", "group", "n_pos", "n_neg", "recall", "fpr", "f1"]].to_string(index=False))
    print("Saved tables to " + str(out))


if __name__ == "__main__":
    main()
