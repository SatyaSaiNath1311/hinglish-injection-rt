# Compare attackers (random, hill-climb, PPO) at equal query budgets on held-out seeds.
#
#   python scripts/run_attackers.py --detector protectai_deberta --judge embedding \
#       --split test --n_seeds 100 --budget 50 --ppo_model results/raw/protectai_deberta_seed42/ppo_attacker
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hinglish_rt.attacker.baselines import hill_climb_attack, ppo_attack, random_attack  # noqa: E402
from hinglish_rt.attacker.reward import EmbeddingIntentJudge, LexicalIntentJudge  # noqa: E402
from hinglish_rt.detectors.registry import build_detectors  # noqa: E402

BUDGETS = [5, 10, 20, 30, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "master.csv"))
    ap.add_argument("--detector", required=True)
    ap.add_argument("--judge", choices=["lexical", "embedding"], default="embedding")
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--categories", nargs="+", default=["override", "exfiltration", "roleplay"],
                    help="seed categories ('all' = no filter); excludes ambiguous 'other' by default")
    ap.add_argument("--min_seed_score", type=float, default=0.9,
                    help="only attack seeds the detector flags with at least this score")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _strict")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--ppo_model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    env_kw = {k: v for k, v in cfg["attacker"]["env"].items() if k != "max_steps"}
    max_steps = cfg["attacker"]["env"]["max_steps"]
    tau = cfg["attacker"]["env"]["tau"]
    det = build_detectors(cfg, only=[a.detector], cache_dir=ROOT / cfg["paths"]["cache_dir"])[a.detector]
    judge = LexicalIntentJudge() if a.judge == "lexical" else EmbeddingIntentJudge()

    df = pd.read_csv(a.data)
    sub = df[(df.split.isin(a.split)) & (df.label == 1) & (df.lang == "en")]
    if "all" not in a.categories:
        sub = sub[sub.category.isin(a.categories)]
    pool = sub.text.tolist()
    seed_score = dict(zip(pool, det.score_batch(pool)))
    cut = max(det.threshold, a.min_seed_score)
    caught = [t for t in pool if seed_score[t] >= cut]
    rng = random.Random(a.seed)
    seeds = rng.sample(caught, min(a.n_seeds, len(caught)))
    print("Seeds in pool: %d | flagged with score >= %.2f: %d -> using %d" %
          (len(pool), cut, len(caught), len(seeds)))

    ppo = None
    if a.ppo_model:
        from stable_baselines3 import PPO  # lazy import
        ppo = PPO.load(a.ppo_model)

    rows = []
    for k, text in enumerate(seeds):
        runs = {
            "random": random_attack(det, judge, text, a.budget, random.Random(a.seed + k), tau, max_steps),
            "hill_climb": hill_climb_attack(det, judge, text, a.budget, random.Random(a.seed + k), tau),
        }
        if ppo is not None:
            runs["ppo"] = ppo_attack(ppo, det, judge, text, a.budget,
                                     dict(env_kw, max_steps=max_steps), seed=a.seed + k)
        for m, r in runs.items():
            rows.append({"detector": a.detector, "method": m, "seed_text": text,
                         "seed_score": seed_score[text],
                         "adv_text": r["text"], **{k2: r[k2] for k2 in ("success", "queries", "score", "intent")}})
    det.flush()

    res = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    res.to_csv(raw / ("attack_runs_" + a.detector + a.tag + ".csv"), index=False, encoding="utf-8")

    summ = []
    for m, g in res.groupby("method"):
        for b in BUDGETS:
            if b <= a.budget:
                summ.append({"detector": a.detector, "tag": a.tag, "method": m, "budget": b, "n": len(g),
                             "asr": float(((g.success) & (g.queries <= b)).mean())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / ("attack_efficiency_" + a.detector + a.tag + ".csv"), index=False)
    res["bucket"] = pd.cut(res.seed_score, [0, 0.9, 0.99, 1.01], labels=["<0.9", "0.9-0.99", ">=0.99"],
                           right=False)
    res["win"] = res.success & (res.queries <= a.budget)
    by = res.groupby(["method", "bucket"], observed=True).win.agg(["mean", "size"]).reset_index()
    by.columns = ["method", "seed_score_bucket", "asr", "n"]
    by.to_csv(out / ("attack_by_seed_score_" + a.detector + a.tag + ".csv"), index=False)
    print(by.round(3).to_string(index=False))
    print(summ.pivot(index="budget", columns="method", values="asr").round(3).to_string())


if __name__ == "__main__":
    main()
