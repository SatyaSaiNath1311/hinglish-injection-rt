# Merge externally generated Hinglish (e.g. LLM-assisted or human-written) into master.csv.
# Input CSV columns: group_id, text[, mix_level][, gen_method]. group_id must exist in master.
# The new rows inherit label, category, source and split from the English row of that group.
#
#   python scripts/import_hinglish.py --csv my_hinglish.csv
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from hinglish_rt.cmi import cmi, script_of  # noqa: E402
from hinglish_rt.data.splits import verify_splits  # noqa: E402
from hinglish_rt.data.text_utils import clean_text, has_trigger_words, normalize_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--master", default=str(ROOT / "data" / "processed" / "master.csv"))
    a = ap.parse_args()

    m = pd.read_csv(a.master)
    new = pd.read_csv(a.csv)
    roots = m[m.lang == "en"].set_index("group_id")
    seen = set(m.text.map(normalize_key))
    rows, skipped = [], 0
    for i, r in new.iterrows():
        gid, text = r["group_id"], clean_text(r["text"])
        if gid not in roots.index or normalize_key(text) in seen or not text:
            skipped += 1
            continue
        root = roots.loc[gid]
        level = r.get("mix_level", "external")
        level = "external" if pd.isna(level) else level
        gm = r.get("gen_method", "external")
        gm = "external" if pd.isna(gm) else gm
        rows.append({"id": "%s-ext%d" % (gid, i), "group_id": gid, "text": text,
                     "label": int(root.label), "lang": "hinglish", "script": script_of(text),
                     "mix_level": level, "gen_method": gm, "category": root.category,
                     "source": root.source, "cmi": round(cmi(text), 2),
                     "has_trigger_words": has_trigger_words(text),
                     "root_trigger": int(root.get("root_trigger", has_trigger_words(root.text))),
                     "verified": 0,
                     "split": root.split})
        seen.add(normalize_key(text))
    out = pd.concat([m, pd.DataFrame(rows)], ignore_index=True)
    verify_splits(out)
    out.to_csv(a.master, index=False, encoding="utf-8")
    print("Added %d rows, skipped %d. master now has %d rows." % (len(rows), skipped, len(out)))


if __name__ == "__main__":
    main()
