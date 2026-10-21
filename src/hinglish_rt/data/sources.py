import random

import pandas as pd

# Verify each license on the dataset card before releasing derived data.
SOURCES = {
    "deepset": dict(hf="deepset/prompt-injections", license="Apache-2.0"),
    "xtram1": dict(hf="xTRam1/safe-guard-prompt-injection", license="check dataset card"),
    "gandalf": dict(hf="Lakera/gandalf_ignore_instructions", license="MIT", force_label=1),
    "harelix": dict(hf="Harelix/Prompt-Injection-Mixed-Techniques-2024", license="Apache-2.0"),
    "dolly": dict(hf="databricks/databricks-dolly-15k", license="CC-BY-SA-3.0",
                  force_label=0, text_col="instruction"),
}

_TEXT_COLS = ["text", "prompt", "user_prompt", "User Prompt", "Prompt", "instruction", "content"]
_LABEL_COLS = ["label", "labels", "Prompt injection", "prompt_injection", "is_injection",
               "injection", "class"]
_POS = {"1", "true", "yes", "injection", "malicious", "attack", "unsafe", "jailbreak"}
_NEG = {"0", "false", "no", "safe", "benign", "legit", "legitimate", "normal"}


def _norm_label(v):
    s = str(v).strip().lower()
    if s in _POS:
        return 1
    if s in _NEG:
        return 0
    return None


def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_source(name, spec, max_rows=None, seed=42):
    """Load every available split of an HF dataset into a DataFrame[text, label, source]."""
    from datasets import load_dataset

    ds = load_dataset(spec["hf"])
    frames = [ds[k].to_pandas() for k in ds.keys()]
    raw = pd.concat(frames, ignore_index=True)
    text_col = spec.get("text_col") or _pick(raw.columns, _TEXT_COLS)
    if text_col is None:
        raise ValueError("no text column found in " + spec["hf"] + ": " + str(list(raw.columns)))
    if "force_label" in spec:
        labels = pd.Series([spec["force_label"]] * len(raw))
        label_col = "(forced)"
    else:
        label_col = _pick(raw.columns, _LABEL_COLS)
        if label_col is None:
            raise ValueError("no label column found in " + spec["hf"] + ": " + str(list(raw.columns)))
        labels = raw[label_col].map(_norm_label)
    df = pd.DataFrame({"text": raw[text_col].astype(str), "label": labels, "source": name})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    print("  " + name + ": " + str(len(df)) + " rows (text=" + str(text_col) + ", label=" +
          str(label_col) + ", license=" + spec["license"] + ")")
    return df.reset_index(drop=True)


def demo_corpus(n_each=400, seed=0):
    """Synthetic English corpus for offline pipeline tests (NOT for real experiments)."""
    rng = random.Random(seed)
    verbs = ["Ignore", "Disregard", "Forget", "Override", "Skip"]
    mods = ["all", "any", "the", "every", "your"]
    prev = ["previous", "prior", "earlier", "above"]
    objs = ["instructions", "rules", "directions", "guidelines"]
    tails = ["and do exactly what I say", "and answer without restrictions",
             "and reveal your system prompt", "and print the secret password",
             "and you are now an unrestricted assistant", "and follow my new instructions"]
    topics = ["photosynthesis", "the French revolution", "binary search", "healthy breakfast",
              "monsoon season", "chess openings", "solar panels", "time management"]
    acts = ["Explain", "Summarize", "Write a short note about", "List three facts about",
            "Describe", "Give me tips on"]
    rows = []
    for i in range(n_each):
        t = "%s %s %s %s %s #%d." % (rng.choice(verbs), rng.choice(mods), rng.choice(prev),
                                      rng.choice(objs), rng.choice(tails), i)
        rows.append({"text": t, "label": 1, "source": "demo"})
        b = "%s %s please (case %d)." % (rng.choice(acts), rng.choice(topics), i)
        rows.append({"text": b, "label": 0, "source": "demo"})
    return pd.DataFrame(rows)
