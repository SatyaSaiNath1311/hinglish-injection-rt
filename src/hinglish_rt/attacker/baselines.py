"""Non-RL attackers sharing the RL attacker's action space, intent judge, tau and query
accounting (one query = one candidate sent to the detector)."""
import random

from ..attacker.actions import ACTION_NAMES, apply_action
from .env import EvasionEnv


def _result(success, queries, text, score, intent):
    return {"success": bool(success), "queries": int(queries), "text": text,
            "score": float(score), "intent": float(intent)}


def _eval(det, judge, original, cand):
    s = float(det.score_batch([cand])[0])
    i = float(judge(original, cand))
    return s, i


def random_attack(det, judge, text, budget, rng, tau, max_steps=8):
    """Random walk over edits; restarts from the original every max_steps."""
    q, cur, steps = 0, text, 0
    best = (1.0, 0.0, text)
    while q < budget:
        if steps >= max_steps:
            cur, steps = text, 0
        cur, _ = apply_action(rng.randrange(len(ACTION_NAMES)), cur, rng)
        steps += 1
        q += 1
        s, i = _eval(det, judge, text, cur)
        if s < det.threshold and i >= tau:
            return _result(True, q, cur, s, i)
        if s < best[0] and i >= tau:
            best = (s, i, cur)
    return _result(False, q, best[2], best[0], best[1])


def hill_climb_attack(det, judge, text, budget, rng, tau):
    """Greedy: try every edit once per round, move to the lowest-score candidate that keeps
    intent >= tau; restart from the original when stuck."""
    q, cur = 0, text
    cur_score = float(det.score_batch([text])[0])
    best = (cur_score, 1.0, text)
    while q < budget:
        round_best, spent = None, 0
        for idx in range(len(ACTION_NAMES)):
            if q >= budget:
                break
            cand, changed = apply_action(idx, cur, rng)
            if not changed:
                continue
            q += 1
            spent += 1
            s, i = _eval(det, judge, text, cand)
            if s < det.threshold and i >= tau:
                return _result(True, q, cand, s, i)
            if i >= tau and (round_best is None or s < round_best[0]):
                round_best = (s, i, cand)
        if spent == 0:
            break
        if round_best and round_best[0] < cur_score - 1e-9:
            cur_score, cur = round_best[0], round_best[2]
            if cur_score < best[0]:
                best = round_best
        else:
            cur, cur_score = text, float(det.score_batch([text])[0])
    return _result(False, q, best[2], best[0], best[1])


def ppo_attack(model, det, judge, text, budget, env_kwargs, seed=0):
    """Roll out a trained policy (anything with .predict(obs, deterministic)) from one seed text,
    restarting episodes until the query budget is spent."""
    env = EvasionEnv(det, judge, [text], seed=seed, **env_kwargs)
    q, best = 0, (1.0, 0.0, text)
    while q < budget:
        obs, _ = env.reset()
        done = False
        while not done and q < budget:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(action)
            q += 1
            done = term or trunc
            if info["success"]:
                return _result(True, q, info["text"], info["score"], info["intent"])
            if info["score"] < best[0] and info["intent"] >= env.tau:
                best = (info["score"], info["intent"], info["text"])
    return _result(False, q, best[2], best[0], best[1])
