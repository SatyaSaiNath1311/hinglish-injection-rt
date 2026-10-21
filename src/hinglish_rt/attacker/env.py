import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..cmi import cmi, script_of
from .actions import ACTION_NAMES, apply_action
from .reward import compute_reward


class EvasionEnv(gym.Env):
    """Black-box evasion environment.

    Episode: start from a malicious seed text that the detector currently catches, apply up to
    `max_steps` edits, stop early on success (score below threshold AND intent >= tau).
    Observation (6 floats in [0, 1]): detector score, CMI/100, length/50, step fraction,
    is-Devanagari flag, intent score.
    """

    metadata = {"render_modes": []}

    def __init__(self, detector, intent_judge, seed_texts, max_steps=8, alpha=1.0, beta=0.5,
                 gamma=0.05, tau=0.7, success_bonus=2.0, seed=0):
        super().__init__()
        self.detector = detector
        self.judge = intent_judge
        self.seed_texts = list(seed_texts)
        if not self.seed_texts:
            raise ValueError("seed_texts is empty")
        self.max_steps = max_steps
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.tau, self.success_bonus = tau, success_bonus
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self.rng = random.Random(seed)
        self.queries = 0
        self.orig = ""
        self.text = ""
        self.score = 0.0
        self.intent = 1.0
        self.t = 0

    def _score(self, text):
        self.queries += 1
        return float(self.detector.score_batch([text])[0])

    def _obs(self):
        v = [
            self.score,
            cmi(self.text) / 100.0,
            min(len(self.text.split()) / 50.0, 1.0),
            self.t / self.max_steps,
            1.0 if script_of(self.text) == "devanagari" else 0.0,
            self.intent,
        ]
        return np.clip(np.array(v, dtype=np.float32), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        self.orig = self.rng.choice(self.seed_texts)
        self.text = self.orig
        self.t = 0
        self.intent = 1.0
        self.score = self._score(self.text)
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        new, changed = apply_action(int(action), self.text, self.rng)
        self.text = new
        self.score = self._score(new)
        self.intent = float(self.judge(self.orig, new))
        success = self.score < self.detector.threshold and self.intent >= self.tau
        reward = compute_reward(
            self.score, self.intent, 1.0, success,
            self.alpha, self.beta, self.gamma, self.success_bonus,
        )
        terminated = bool(success)
        truncated = (not success) and self.t >= self.max_steps
        info = {
            "success": bool(success),
            "changed": bool(changed),
            "score": self.score,
            "intent": self.intent,
            "original": self.orig,
            "text": self.text,
            "queries": self.queries,
        }
        return self._obs(), float(reward), terminated, truncated, info
