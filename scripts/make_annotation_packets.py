# Create independent annotation packets (two annotators) from annotation_sample.csv.
#
#   python scripts/make_annotation_packets.py
# Output: data/annotation/annotator_A.csv, annotator_B.csv  (different row order, labels hidden)
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "data" / "processed" / "annotation_sample.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "annotation"))
    a = ap.parse_args()
    s = pd.read_csv(a.sample, keep_default_na=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, seed in (("A", 1), ("B", 2)):
        p = s[["id", "english_text", "text"]].sample(frac=1, random_state=seed).copy()
        p["intent_preserved_0_1"] = ""
        p["fluency_1_5"] = ""
        p["notes"] = ""
        p.to_csv(out / ("annotator_" + name + ".csv"), index=False, encoding="utf-8")
    print("Wrote %d rows to annotator_A.csv and annotator_B.csv in %s" % (len(s), out))
    print("Give each annotator ONE file plus docs/ANNOTATION_GUIDE.md. Do not let them compare.")


if __name__ == "__main__":
    main()
