# End-to-end sanity check. CPU only, no model downloads.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hinglish_rt.attacker.env import EvasionEnv  # noqa: E402
from hinglish_rt.attacker.reward import LexicalIntentJudge  # noqa: E402
from hinglish_rt.cmi import cmi  # noqa: E402
from hinglish_rt.detectors.base import CachedDetector  # noqa: E402
from hinglish_rt.detectors.regex_detector import RegexDetector  # noqa: E402
from hinglish_rt.eval.metrics import summarize  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
    df = pd.read_csv(ROOT / "data" / "gold" / "seed.csv")

    det = CachedDetector(RegexDetector())
    df["score"] = det.score_batch(df.text.tolist())
    df["cmi"] = df.text.map(cmi).round(1)
    print(df[["id", "lang", "label", "score", "cmi"]].to_string(index=False))

    print("\nPer-language metrics (regex baseline):")
    for lang, g in df.groupby("lang"):
        print(lang, summarize(g.label, g.score, det.threshold))

    seeds = [
        t for t in df[(df.label == 1) & (df.lang == "en")].text
        if det.score_batch([t])[0] >= det.threshold
    ]
    env = EvasionEnv(det, LexicalIntentJudge(), seeds, **cfg["attacker"]["env"])
    wins, n_ep = 0, 20
    for ep in range(n_ep):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        wins += int(info["success"])
    print("\nRandom-policy evasions: " + str(wins) + "/" + str(n_ep) +
          " (queries: " + str(env.queries) + ")")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
