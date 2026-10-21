import json
import random
from pathlib import Path

import pandas as pd

from ..cmi import cmi, script_of
from .hinglish import make_variants, mt_translate_devanagari
from .manual_pairs import GOLD_MALICIOUS, HARD_NEGATIVES
from .sources import SOURCES, demo_corpus, load_source
from .splits import assign_splits, save_splits, verify_splits
from .text_utils import (categorize, clean_text, has_trigger_words, is_english, near_dedupe,
                         normalize_key, stable_id)

COLUMNS = ["id", "group_id", "text", "label", "lang", "script", "mix_level", "gen_method",
           "category", "source", "cmi", "has_trigger_words", "root_trigger", "verified", "split"]


def _load_all(source_names, cap, seed, local_csv, demo):
    if demo:
        return demo_corpus(seed=seed)
    if local_csv:
        df = pd.read_csv(local_csv)
        if "source" not in df.columns:
            df["source"] = "local"
        return df[["text", "label", "source"]]
    frames = []
    for name in source_names:
        try:
            frames.append(load_source(name, SOURCES[name], max_rows=cap * 3, seed=seed))
        except Exception as e:  # keep going if one source fails
            print("  SKIPPED " + name + ": " + str(e))
    if not frames:
        raise RuntimeError("no source could be loaded")
    return pd.concat(frames, ignore_index=True)


def _english_rows(df, cap, seed, manual_keys):
    n0 = len(df)
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len().between(2, 150)]
    df = df[df["text"].map(is_english)]
    print("  after clean + English filter: %d (from %d)" % (len(df), n0))

    df["_k"] = df["text"].map(normalize_key)
    conflict = df.groupby("_k")["label"].nunique()
    bad = set(conflict[conflict > 1].index)
    df = df[~df["_k"].isin(bad | manual_keys)].drop_duplicates("_k")
    print("  after exact dedupe (+%d label conflicts dropped): %d" % (len(bad), len(df)))

    parts = []
    for (_, _), g in df.groupby(["source", "label"]):
        parts.append(g.sample(n=min(cap, len(g)), random_state=seed))
    df = pd.concat(parts).sort_values("_k").reset_index(drop=True)
    keep = near_dedupe(df["text"].tolist())
    df = df[keep].reset_index(drop=True)
    print("  after per-source cap (%d) + near-dedupe: %d" % (cap, len(df)))
    return df.drop(columns=["_k"])


def _row(**kw):
    base = dict(lang="hinglish", script="roman", mix_level="none", gen_method="original",
                verified=0, cmi=0.0)
    base.update(kw)
    base["has_trigger_words"] = has_trigger_words(base["text"])
    base.setdefault("root_trigger", base["has_trigger_words"])
    return base


def build_master(out_dir="data/processed", sources=("deepset", "xtram1", "gandalf"), cap=2000,
                 seed=42, ood_category="roleplay", use_mt=False, local_csv=None, demo=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading sources...")
    raw = _load_all(sources, cap, seed, local_csv, demo)

    manual_keys = {normalize_key(en) for _, en, _ in GOLD_MALICIOUS} | \
                  {normalize_key(en) for en, _ in HARD_NEGATIVES}
    print("Cleaning...")
    eng = _english_rows(raw, cap, seed, manual_keys)

    rows = []
    for _, r in eng.iterrows():
        gid = stable_id(r["text"])
        rows.append(_row(id=gid + "-en", group_id=gid, text=r["text"], label=int(r["label"]),
                         lang="en", category=categorize(r["text"], r["label"]),
                         source=r["source"], cmi=0.0, verified=1))
        rng = random.Random(str(seed) + ":" + gid)
        for v in make_variants(r["text"], rng):
            rows.append(_row(id=gid + "-" + v["mix_level"], group_id=gid, text=v["text"],
                             label=int(r["label"]), script=v["script"], mix_level=v["mix_level"],
                             gen_method=v["gen_method"], category=categorize(r["text"], r["label"]),
                             source=r["source"], cmi=v["cmi"],
                             root_trigger=has_trigger_words(r["text"])))

    for i, (cat, en, hi) in enumerate(GOLD_MALICIOUS):
        gid = "m_gold_%03d" % i
        rows.append(_row(id=gid + "-en", group_id=gid, text=en, label=1, lang="en", category=cat,
                         source="manual_gold", verified=1))
        rows.append(_row(id=gid + "-manual", group_id=gid, text=hi, label=1, mix_level="manual",
                         gen_method="manual", category=cat, source="manual_gold", cmi=round(cmi(hi), 2),
                         root_trigger=has_trigger_words(en)))
    for i, (en, hi) in enumerate(HARD_NEGATIVES):
        gid = "m_hneg_%03d" % i
        rows.append(_row(id=gid + "-en", group_id=gid, text=en, label=0, lang="en",
                         category="hard_negative", source="manual_hard_negative", verified=1))
        rows.append(_row(id=gid + "-manual", group_id=gid, text=hi, label=0, mix_level="manual",
                         gen_method="manual", category="hard_negative",
                         source="manual_hard_negative", cmi=round(cmi(hi), 2),
                         root_trigger=has_trigger_words(en)))

    if use_mt:
        print("Machine-translating to Devanagari (optional track)...")
        en_rows = [r for r in rows if r["lang"] == "en" and not r["source"].startswith("manual_")]
        hi_texts = mt_translate_devanagari([r["text"] for r in en_rows])
        for r, t in zip(en_rows, hi_texts):
            if script_of(t) == "devanagari" and t.strip():
                rows.append(_row(id=r["group_id"] + "-mt", group_id=r["group_id"], text=t.strip(),
                                 label=r["label"], script="devanagari", mix_level="deva_full",
                                 gen_method="mt", category=r["category"], source=r["source"],
                                 cmi=round(cmi(t), 2), root_trigger=r["root_trigger"]))

    df = pd.DataFrame(rows)
    # Drop duplicate texts globally (keep first), and rows whose text collides across groups.
    df["_k"] = df["text"].map(normalize_key)
    df = df.drop_duplicates("_k", keep="first").drop(columns=["_k"])
    # Every group must keep its English root.
    roots = set(df[df["lang"] == "en"]["group_id"])
    df = df[df["group_id"].isin(roots)].reset_index(drop=True)

    ratios = (0.70, 0.15, 0.15)
    if ood_category and ((df["category"] == ood_category) & (df["label"] == 1)
                         & (df["lang"] == "en")).sum() < 30:
        print("  OOD category '" + str(ood_category) + "' too small; disabling test_ood")
        ood_category = None
    df, mapping = assign_splits(df, seed, ratios, ood_category)
    verify_splits(df)
    df = df[COLUMNS]

    df.to_csv(out_dir / "master.csv", index=False, encoding="utf-8")
    save_splits(mapping, out_dir / "splits.json", seed, ratios, ood_category)
    _write_annotation_sample(df, out_dir / "annotation_sample.csv", seed)
    _write_card(df, out_dir / "dataset_card.md", sources, cap, seed, ood_category, use_mt, demo)
    print("Saved master (%d rows) to %s" % (len(df), out_dir))
    return df


def _write_annotation_sample(df, path, seed, n=300):
    hi = df[df["lang"] == "hinglish"].copy()
    en = df[df["lang"] == "en"].set_index("group_id")["text"]
    hi["english_text"] = hi["group_id"].map(en)
    parts = []
    strata = hi.groupby(["label", "mix_level"])
    per = max(1, n // max(1, strata.ngroups))
    for _, g in strata:
        parts.append(g.sample(n=min(per, len(g)), random_state=seed))
    s = pd.concat(parts).head(n)
    s = s[["id", "group_id", "english_text", "text", "label", "mix_level"]].copy()
    s["intent_preserved_0_1"] = ""
    s["fluency_1_5"] = ""
    s["annotator"] = ""
    s["notes"] = ""
    s.to_csv(path, index=False, encoding="utf-8")


def _write_card(df, path, sources, cap, seed, ood, use_mt, demo):
    tab = df.groupby(["split", "label", "lang"]).size().unstack("lang", fill_value=0)
    lines = [
        "# Dataset card: Hinglish prompt-injection benchmark", "",
        "Generated by scripts/build_dataset.py (seed=%d, per-source cap=%d)." % (seed, cap),
        "Sources: " + ("DEMO synthetic corpus (not for experiments)" if demo else ", ".join(sources)),
        "Held-out OOD category: " + str(ood), "",
        "## Schema",
        "id, group_id (pair id shared by an English prompt and all its variants), text, label "
        "(1 = injection), lang, script, mix_level, gen_method, category, source, cmi, "
        "has_trigger_words, verified, split.", "",
        "## Splits", "Deterministic hash of group_id (splits.json). A group never spans splits. "
        "'gold' = author-written fresh pairs, never used for training or RL seeds. "
        "'test_ood' = injections of one held-out category.", "",
        "## Counts (rows)", "", "```", tab.to_string(), "```", "",
        "## Generation methods",
        "- original: public English data, cleaned, language-filtered, deduplicated.",
        "- lexicon: rule-based Hinglish (phrase rules + word table, light/medium/heavy, mixed "
        "Devanagari). Approximation: no verb conjugation or Hindi word order.",
        "- trig_only / nontrig_only: controlled ablation. trig_only swaps ONLY English trigger "
        "words (ignore, instructions, previous, rules...) for Hindi; nontrig_only swaps everything "
        "else but keeps the trigger words in English. root_trigger = trigger flag of the English "
        "source prompt, inherited by all its variants.",
        "- manual: author-written Hinglish twins (gold set, hard negatives).",
        "- mt: optional machine translation to Devanagari" + (" (included)." if use_mt else " (not included)."),
        "- external: Hinglish imported from another tool (scripts/import_hinglish.py).", "",
        "## Verification status",
        "verified=1 only for public/original English rows. All Hinglish rows are verified=0 until "
        "bilingual annotators review annotation_sample.csv (report Cohen's kappa).", "",
        "## Limitations",
        "- Lexicon Hinglish is synthetic and less varied than natural code-mixing.",
        "- Public sources may overlap with detector training data (contamination); use the gold "
        "split for contamination-free comparison.",
        "- Check every source license before redistributing derived data.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
