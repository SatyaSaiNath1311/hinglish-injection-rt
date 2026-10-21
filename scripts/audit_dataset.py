# Data audit + Phase 1 exit checklist. Run after build_dataset.py.
#
#   python scripts/audit_dataset.py
# Writes: results/tables/data_audit.md (+ CSVs), data/annotation/label_audit_sample.csv
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hinglish_rt.data.splits import verify_splits  # noqa: E402
from hinglish_rt.data.text_utils import normalize_key  # noqa: E402

PUBLIC = ~pd.Series(dtype=bool)


def md(title, df, index=True):
    return "\n## " + title + "\n\n```\n" + df.to_string(index=index) + "\n```\n"


def cross_split_near_dups(en, thr=0.9, chunk=800):
    from sklearn.feature_extraction.text import TfidfVectorizer

    clean = lambda t: re.sub(r"[^\w\s]", "", normalize_key(t))
    tr = en[en.split == "train"]
    ot = en[en.split != "train"]
    if tr.empty or ot.empty:
        return pd.DataFrame()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(pd.concat([tr.text, ot.text]).map(clean))
    Xtr, Xot = vec.transform(tr.text.map(clean)), vec.transform(ot.text.map(clean))
    rows = []
    for s in range(0, len(ot), chunk):
        sims = (Xot[s:s + chunk] @ Xtr.T).toarray()
        j, m = sims.argmax(1), sims.max(1)
        for k in np.where(m > thr)[0]:
            rows.append({"other_id": ot.iloc[s + k].id, "other_split": ot.iloc[s + k].split,
                         "train_id": tr.iloc[j[k]].id, "similarity": round(float(m[k]), 3),
                         "other_text": ot.iloc[s + k].text, "train_text": tr.iloc[j[k]].text})
    return pd.DataFrame(rows)


def label_audit_sample(en, seed, out):
    pub = en[~en.source.str.startswith(("manual_", "synthetic_"))]
    mal, ben = pub[pub.label == 1], pub[pub.label == 0]
    parts = []
    for cat, n in (("other", 40), ("override", 20), ("exfiltration", 20), ("roleplay", 20)):
        g = mal[mal.category == cat]
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    bt = ben[ben.has_trigger_words == 1]
    parts.append(bt.sample(n=min(50, len(bt)), random_state=seed))
    bn = ben[ben.has_trigger_words == 0]
    parts.append(bn.sample(n=min(50, len(bn)), random_state=seed))
    s = pd.concat(parts)[["id", "text", "label", "source", "category"]].rename(
        columns={"label": "current_label"})
    s = s.sample(frac=1, random_state=seed).reset_index(drop=True)
    s["human_label"] = ""   # malicious / benign / ambiguous
    s["notes"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False, encoding="utf-8")
    return len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "master.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(a.data, keep_default_na=False)
    d["label"] = d["label"].astype(int)
    d["cmi"] = pd.to_numeric(d["cmi"], errors="coerce").fillna(0.0)
    d["has_trigger_words"] = d["has_trigger_words"].astype(int)
    en, hi = d[d.lang == "en"].copy(), d[d.lang == "hinglish"].copy()
    en["words"] = en.text.str.split().str.len()
    rep = ["# Data audit\n", "Source file: " + str(a.data)]

    # 1. size and balance
    size = d.groupby(["split", "label", "lang"]).size().unstack("lang", fill_value=0)
    groups = d.groupby(["split", "label"]).group_id.nunique().unstack("label", fill_value=0)
    rep.append(md("Rows by split / label / language", size))
    rep.append(md("Independent prompts (groups) by split / label  (the real sample size)", groups))
    size.to_csv(out / "audit_size.csv")

    # 2. twin coverage
    en_groups = en.groupby("label").group_id.nunique()
    cov = []
    for lvl, g in hi.groupby("mix_level"):
        for lab in (0, 1):
            n = g[g.label == lab].group_id.nunique()
            cov.append({"mix_level": lvl, "label": lab, "groups_with_variant": n,
                        "coverage": round(n / max(1, en_groups.get(lab, 1)), 3)})
    cov = pd.DataFrame(cov)
    any_cov = hi.groupby("label").group_id.nunique() / en_groups
    rep.append(md("Hinglish twin coverage (share of English prompts that have each variant)", cov, False))
    rep.append("\nShare of English prompts with at least one Hinglish variant:\n```\n" +
               any_cov.round(3).to_string() + "\n```\n")
    cov.to_csv(out / "audit_twin_coverage.csv", index=False)

    # 3. categories, sources, trigger confound
    rep.append(md("Category by label (English rows)", en.groupby(["category", "label"]).size().unstack(fill_value=0)))
    rep.append(md("Source by label (English rows)", en.groupby(["source", "label"]).size().unstack(fill_value=0)))
    trig = en.groupby("label").has_trigger_words.mean().round(3)
    rep.append("\nShare of English prompts containing English trigger words, by label "
               "(a large gap means 'trigger word' is a shortcut for the label):\n```\n" +
               trig.to_string() + "\n```\n")

    # 4. length and CMI
    rep.append(md("English length (words) by label", en.groupby("label").words.describe().round(1)))
    rep.append(md("CMI by mix_level (Hinglish rows)", hi.groupby("mix_level").cmi.describe().round(1)))
    rep.append(md("Script share (Hinglish rows)", hi.script.value_counts(normalize=True).round(3).to_frame("share")))

    # 5. variant quality outliers
    en_words = en.set_index("group_id").words
    hi["ratio"] = hi.text.str.split().str.len() / hi.group_id.map(en_words)
    outl = hi[(hi.ratio > 2.0) | (hi.ratio < 0.5)][["id", "mix_level", "ratio", "text"]]
    outl.to_csv(out / "audit_variant_outliers.csv", index=False)
    rep.append("\nHinglish variants with word-count ratio vs English outside [0.5, 2.0]: %d (see "
               "audit_variant_outliers.csv)\n" % len(outl))

    # 6. leakage
    leak_ok = True
    try:
        verify_splits(d)
    except AssertionError as e:
        leak_ok = False
        rep.append("\nLEAKAGE: " + str(e) + "\n")
    nd = cross_split_near_dups(en)
    nd.to_csv(out / "audit_cross_split_near_dups.csv", index=False)
    nonpublic_other = en[en.split != "train"]
    nd_rate = len(nd) / max(1, len(nonpublic_other))
    rep.append("\nExact/group leakage check: %s\nEnglish prompts outside train with a near-duplicate "
               "(cosine > 0.9) in train: %d (%.1f%% of %d)\n" %
               ("PASS" if leak_ok else "FAIL", len(nd), 100 * nd_rate, len(nonpublic_other)))

    # 7. label-noise sample
    n_sample = label_audit_sample(en, a.seed, ROOT / "data" / "annotation" / "label_audit_sample.csv")

    # 8. checklist
    gold_m = d[d.source == "manual_gold"].group_id.nunique()
    gold_b = d[d.source == "manual_hard_negative"].group_id.nunique()
    hardneg = en[en.source == "synthetic_hard_negative"].shape[0]
    pubmal = en[(en.label == 1) & ~en.source.str.startswith(("manual_", "synthetic_"))]
    other_share = float((pubmal.category == "other").mean()) if len(pubmal) else 0.0
    med = cov[(cov.mix_level == "medium")].coverage.min() if (cov.mix_level == "medium").any() else 0.0
    audit_done = ROOT / "data" / "annotation" / "label_audit_done.csv"
    kappa_file = out / "annotation_summary.csv"
    status = []

    def add(name, ok, detail):
        status.append((name, "PASS" if ok else "FAIL", detail))

    def pending(name, detail):
        status.append((name, "PENDING", detail))

    add("No exact or group leakage across splits", leak_ok, "")
    add("Near-duplicate rate across splits <= 2%", nd_rate <= 0.02, "%.1f%%" % (100 * nd_rate))
    add("Gold malicious pairs >= 80 (95% CI half-width ~0.09 at recall 0.8)", gold_m >= 80, str(gold_m))
    add("Gold benign hard-negative pairs >= 80", gold_b >= 80, str(gold_b))
    add("Synthetic hard-negative prompts >= 300", hardneg >= 300, str(hardneg))
    add("Public injections in ambiguous 'other' category <= 10%", other_share <= 0.10,
        "%.1f%%" % (100 * other_share))
    add("Medium-mix Hinglish coverage >= 90% for both labels", med >= 0.90, "%.1f%%" % (100 * med))
    if audit_done.exists():
        done = pd.read_csv(audit_done, keep_default_na=False)
        n_done = int((done.human_label.str.strip() != "").sum())
        add("Label audit completed (>= 150 rows)", n_done >= 150, str(n_done))
    else:
        pending("Label audit completed (>= 150 rows)", "label_audit_done.csv not found")
    if kappa_file.exists():
        k = pd.read_csv(kappa_file).set_index("metric").value
        kv, ac1, agr = float(k["kappa_intent"]), float(k["gwet_ac1_intent"]), float(k["agreement_intent"])
        # Pre-declared rule: kappa >= 0.6, or (agreement >= 0.90 and AC1 >= 0.80) when one label
        # dominates and kappa is unreliable.
        add("Annotator agreement on intent (kappa >= 0.6, or agreement >= 0.9 and AC1 >= 0.8)",
            kv >= 0.6 or (agr >= 0.9 and ac1 >= 0.8),
            "kappa=%.2f AC1=%.2f agreement=%.2f" % (kv, ac1, agr))
    else:
        pending("Annotator agreement on intent (kappa >= 0.6, or agreement >= 0.9 and AC1 >= 0.8)",
                "annotation_summary.csv not found")

    chk = pd.DataFrame(status, columns=["check", "status", "detail"])
    rep.append(md("PHASE 1 EXIT CHECKLIST", chk, False))
    (out / "data_audit.md").write_text("\n".join(rep), encoding="utf-8")
    chk.to_csv(out / "audit_checklist.csv", index=False)

    print(chk.to_string(index=False))
    print("\nWrote %s (label audit sample: %d rows in data/annotation/)" %
          (out / "data_audit.md", n_sample))


if __name__ == "__main__":
    main()
