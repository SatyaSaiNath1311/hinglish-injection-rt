import re

from ..cmi import script_of
from ..data.hinglish import apply_one_phrase
from ..lexicon import EN_TO_HI_DEVA, EN_TO_HI_ROMAN, FILLERS

# Edit operations available to every attacker (random, hill-climbing, PPO). Add more here
# (clause reordering, spelling variants, mixed-script words...); sizes update automatically.
ACTION_NAMES = ["to_roman_hindi", "to_devanagari", "phrase_roman", "phrase_deva",
                "insert_filler", "drop_punct"]
_TRAIL = ".,!?;:"


def _swap_word(text, table, rng):
    words = text.split()
    cands = [i for i, w in enumerate(words) if w.rstrip(_TRAIL).lower() in table]
    if not cands:
        return text, False
    i = rng.choice(cands)
    core = words[i].rstrip(_TRAIL)
    trail = words[i][len(core):]
    words[i] = rng.choice(table[core.lower()]) + trail
    new = " ".join(" ".join(words).split())
    return new, new != text


def _insert_filler(text, rng):
    words = text.split()
    words.insert(rng.randint(0, len(words)), rng.choice(FILLERS))
    return " ".join(words), True


def _drop_punct(text):
    new = re.sub(r"[.,!?;:]", "", text)
    return new, new != text


def apply_action(idx, text, rng):
    """Apply edit `idx` to `text`. Returns (new_text, changed)."""
    name = ACTION_NAMES[idx]
    if name == "to_roman_hindi":
        return _swap_word(text, EN_TO_HI_ROMAN, rng)
    if name == "to_devanagari":
        return _swap_word(text, EN_TO_HI_DEVA, rng)
    if name == "phrase_roman":
        return apply_one_phrase(text, "roman", rng)
    if name == "phrase_deva":
        return apply_one_phrase(text, "deva", rng)
    if name == "insert_filler":
        return _insert_filler(text, rng)
    if name == "drop_punct":
        return _drop_punct(text)
    raise ValueError("Unknown action index: " + str(idx))
