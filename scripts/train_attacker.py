# Train a PPO edit-action attacker on TRAIN seeds, evaluate on held-out TEST seeds.
#
#   python scripts/train_attacker.py --detector regex_baseline --judge lexical --timesteps 2000
#   python scripts/train_attacker.py --detector protectai_deberta --judge embedding --timesteps 50000
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hinglish_rt.attacker.env import EvasionEnv  # noqa: E402
from hinglish_rt.attacker.reward import EmbeddingIntentJudge, LexicalIntentJudge  # noqa: E402
from hinglish_rt.detectors.registry import build_detectors  # noqa: E402


def _caught_seeds(df, splits, det, categories, min_score):
    sub = df[(df.split.isin(splits)) & (df.label == 1) & (df.lang == "en")]
    if "all" not in categories:
        sub = sub[sub.category.isin(categories)]
    pool = sub.text.tolist()
    cut = max(det.threshold, min_score)
    return [t for t, s in zip(pool, det.score_batch(pool)) if s >= cut]


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=False):
        return self.env.action_space.sample(), None


def _rollouts(env, model, n, seed0):
    rows, wins = [], 0
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        if info["success"]:
            wins += 1
            rows.append({k: info[k] for k in ("original", "text", "score", "intent")})
    return wins, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "master.csv"))
    ap.add_argument("--detector", required=True)
    ap.add_argument("--judge", choices=["lexical", "embedding"], default="embedding")
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--categories", nargs="+", default=["override", "exfiltration", "roleplay"])
    ap.add_argument("--min_seed_score", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
    a = ap.parse_args()

    from stable_baselines3 import PPO  # lazy import

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = pd.read_csv(a.data)
    det = build_detectors(cfg, only=[a.detector], cache_dir=ROOT / cfg["paths"]["cache_dir"])[a.detector]
    judge = LexicalIntentJudge() if a.judge == "lexical" else EmbeddingIntentJudge()

    train_seeds = _caught_seeds(df, ["train"], det, a.categories, a.min_seed_score)
    test_seeds = _caught_seeds(df, ["test"], det, a.categories, a.min_seed_score)
    print("Seeds caught: train=%d, test=%d" % (len(train_seeds), len(test_seeds)))

    env = EvasionEnv(det, judge, train_seeds, seed=a.seed, **cfg["attacker"]["env"])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=1, **cfg["attacker"]["ppo"])
    model.learn(total_timesteps=a.timesteps)

    out_dir = Path(a.out) / (a.detector + "_seed" + str(a.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "ppo_attacker"))

    summary = []
    for name, seeds in (("train_seeds", train_seeds), ("test_seeds", test_seeds)):
        e = EvasionEnv(det, judge, seeds, seed=a.seed + 1, **cfg["attacker"]["env"])
        for policy_name, policy in (("ppo", model), ("random_policy", RandomPolicy(e))):
            wins, rows = _rollouts(e, policy, a.eval_episodes, 10_000)
            if policy_name == "ppo":
                pd.DataFrame(rows).to_csv(out_dir / ("evasions_" + name + ".csv"), index=False,
                                          encoding="utf-8")
            summary.append({"detector": a.detector, "seeds": name, "policy": policy_name,
                            "episodes": a.eval_episodes, "success_rate": wins / a.eval_episodes})
            print(name + " / " + policy_name + ": " + str(wins) + "/" + str(a.eval_episodes))
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    det.flush()


if __name__ == "__main__":
    main()
