# Estimate label noise from the human-labelled audit sample and optionally remove noisy prompts.
#
#   python scripts/apply_label_audit.py                  # report only
#   python scripts/apply_label_audit.py --apply          # also remove prompts whose label a human rejected
# Input: data/annotation/label_audit_done.csv (human_label = malicious / benign / ambiguous)
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "annotation" / "label_audit_done.csv"))
    ap.add_argument("--master", default=str(ROOT / "data" / "processed" / "master.csv"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    d = pd.read_csv(a.csv, keep_default_na=False)
    d["human_label"] = d["human_label"].str.strip().str.lower()
    d = d[d.human_label.isin(["malicious", "benign", "ambiguous"])].copy()
    want = d.current_label.map({1: "malicious", 0: "benign"})
    d["noisy"] = (d.human_label != want).astype(int)
    print("Labelled rows: %d | label disagreements: %d" % (len(d), d.noisy.sum()))
    rows = []
    for (lab, cat), g in d.groupby(["current_label", "category"]):
        lo, hi = wilson(int(g.noisy.sum()), len(g))
        rows.append({"current_label": lab, "category": cat, "n": len(g),
                     "noisy": int(g.noisy.sum()), "rate": round(g.noisy.mean(), 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "label_noise_by_category.csv", index=False)
    print("\nUse these rates in the paper's limitations section. If 'other' is much noisier than the "
          "rest, keep it out of attack seeds (the attackers already do by default).")

    if a.apply:
        m = pd.read_csv(a.master, keep_default_na=False)
        gid = m.set_index("id").group_id
        bad_groups = {gid[i] for i in d[d.noisy == 1].id if i in gid.index}
        before = len(m)
        m = m[~m.group_id.isin(bad_groups)]
        m.to_csv(a.master, index=False, encoding="utf-8")
        pd.Series(sorted(bad_groups), name="group_id").to_csv(
            ROOT / "data" / "annotation" / "removed_groups.csv", index=False)
        print("Removed %d groups (%d rows)." % (len(bad_groups), before - len(m)))


if __name__ == "__main__":
    main()
