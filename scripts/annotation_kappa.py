# Agreement statistics for two completed annotation packets, plus approved/rejected lists.
#
#   python scripts/annotation_kappa.py                         # needs annotator_A_done.csv / _B_done.csv
#   python scripts/annotation_kappa.py --apply                 # also mark verified=1 / drop rejected rows
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "data" / "annotation" / "annotator_A_done.csv"))
    ap.add_argument("--b", default=str(ROOT / "data" / "annotation" / "annotator_B_done.csv"))
    ap.add_argument("--master", default=str(ROOT / "data" / "processed" / "master.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    ra = pd.read_csv(a.a, keep_default_na=False)
    rb = pd.read_csv(a.b, keep_default_na=False)
    for r in (ra, rb):
        r["intent"] = pd.to_numeric(r["intent_preserved_0_1"], errors="coerce")
        r["flu"] = pd.to_numeric(r["fluency_1_5"], errors="coerce")
    m = ra[["id", "intent", "flu"]].merge(rb[["id", "intent", "flu"]], on="id", suffixes=("_a", "_b"))
    ok = m.dropna()
    print("Rows in both files: %d | fully annotated by both: %d" % (len(m), len(ok)))
    if len(ok) < 30:
        sys.exit("Too few complete rows to compute agreement (need >= 30).")

    k_int = cohen_kappa_score(ok.intent_a, ok.intent_b)
    k_flu = cohen_kappa_score(ok.flu_a, ok.flu_b, weights="quadratic", labels=[1, 2, 3, 4, 5])
    both1 = float(((ok.intent_a == 1) & (ok.intent_b == 1)).mean())
    pa = float((ok.intent_a == ok.intent_b).mean())
    pi = float((ok.intent_a.mean() + ok.intent_b.mean()) / 2)     # prevalence of label 1
    pe = 2 * pi * (1 - pi)
    ac1 = (pa - pe) / (1 - pe) if pe < 1 else float("nan")        # Gwet's AC1, robust to prevalence
    rows = [("n_rows", len(ok)), ("kappa_intent", k_int), ("gwet_ac1_intent", ac1),
            ("agreement_intent", pa),
            ("kappa_fluency_quadratic", k_flu), ("mean_fluency", float((ok.flu_a + ok.flu_b).mean() / 2)),
            ("intent_preserved_by_both", both1)]
    summ = pd.DataFrame(rows, columns=["metric", "value"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "annotation_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    if np.isnan(k_int):
        print("kappa is undefined because one annotator used a single label; check the packets.")

    master = pd.read_csv(a.master, keep_default_na=False)
    info = master.set_index("id")[["mix_level", "label", "source"]]
    ok = ok.join(info, on="id")
    ok["both_intent"] = ((ok.intent_a == 1) & (ok.intent_b == 1)).astype(int)
    ok["mean_flu"] = (ok.flu_a + ok.flu_b) / 2
    by = ok.groupby("mix_level").agg(n=("id", "size"), intent_preserved_by_both=("both_intent", "mean"),
                                     mean_fluency=("mean_flu", "mean")).round(3)
    by.to_csv(out / "annotation_by_mix_level.csv")
    print("\nBy mix level:\n" + by.to_string())

    approved = ok[(ok.both_intent == 1) & (ok.mean_flu >= 3)].id
    rejected = ok[(ok.intent_a == 0) & (ok.intent_b == 0)].id
    adjud = ok[ok.intent_a != ok.intent_b].id
    adir = ROOT / "data" / "annotation"
    adir.mkdir(parents=True, exist_ok=True)
    approved.to_csv(adir / "approved_ids.csv", index=False)
    rejected.to_csv(adir / "rejected_ids.csv", index=False)
    adjud.to_csv(adir / "needs_adjudication_ids.csv", index=False)
    print("\napproved: %d | rejected (both say meaning lost): %d | need adjudication: %d" %
          (len(approved), len(rejected), len(adjud)))

    if a.apply:
        master.loc[master.id.isin(set(approved)), "verified"] = 1
        before = len(master)
        master = master[~master.id.isin(set(rejected))]
        master.to_csv(a.master, index=False, encoding="utf-8")
        print("Applied: %d rows marked verified=1, %d rows removed." % (len(approved), before - len(master)))


if __name__ == "__main__":
    main()
