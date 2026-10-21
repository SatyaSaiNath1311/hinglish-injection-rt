import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..cmi import cmi, script_of
from ..data.text_utils import count_trigger_words
from .actions import ACTION_NAMES, apply_action
from .reward import compute_reward, score_logit

N_ACT = len(ACTION_NAMES)
OBS_DIM = 8 + 2 * N_ACT


class EvasionEnv(gym.Env):
    """Black-box evasion environment.

    Episode: start from a malicious seed the detector currently flags, apply up to `max_steps`
    edits, stop on success (score below threshold AND intent >= tau).

    Observation (all in [0, 1]), OBS_DIM floats:
      detector log-odds (normalised), detector score, CMI/100, length/50, step fraction,
      is-Devanagari, intent score, trigger-word count/4,
      applicability of each edit (would it change the text?), one-hot of the last action.
    The applicability and trigger features let the policy pick edits that can act on THIS text.
    """

    metadata = {"render_modes": []}

    def __init__(self, detector, intent_judge, seed_texts, max_steps=8, alpha=1.0, beta=2.0,
                 gamma=0.02, tau=0.7, success_bonus=5.0, seed=0):
        super().__init__()
        self.detector = detector
        self.judge = intent_judge
        self.seed_texts = list(seed_texts)
        if not self.seed_texts:
            raise ValueError("seed_texts is empty")
        self.max_steps = max_steps
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.tau, self.success_bonus = tau, success_bonus
        self.action_space = spaces.Discrete(N_ACT)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.rng = random.Random(seed)
        self.queries = 0
        self.orig = ""
        self.text = ""
        self.score = 0.0
        self.logit = 0.0
        self.intent = 1.0
        self.t = 0
        self.last_action = -1

    def _score(self, text):
        self.queries += 1
        return float(self.detector.score_batch([text])[0])

    def _applicable(self):
        return [1.0 if apply_action(i, self.text, random.Random(0))[1] else 0.0
                for i in range(N_ACT)]

    def _obs(self):
        last = [1.0 if i == self.last_action else 0.0 for i in range(N_ACT)]
        v = [
            (self.logit + 16.2) / 32.4,
            self.score,
            cmi(self.text) / 100.0,
            min(len(self.text.split()) / 50.0, 1.0),
            self.t / self.max_steps,
            1.0 if script_of(self.text) == "devanagari" else 0.0,
            self.intent,
            min(count_trigger_words(self.text) / 4.0, 1.0),
        ] + self._applicable() + last
        return np.clip(np.array(v, dtype=np.float32), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        self.orig = self.rng.choice(self.seed_texts)
        self.text = self.orig
        self.t = 0
        self.intent = 1.0
        self.last_action = -1
        self.score = self._score(self.text)
        self.logit = score_logit(self.score)
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        self.last_action = int(action)
        new, changed = apply_action(int(action), self.text, self.rng)
        prev_logit = self.logit
        self.text = new
        if changed:  # an unchanged text needs no new detector query
            self.score = self._score(new)
            self.logit = score_logit(self.score)
            self.intent = float(self.judge(self.orig, new))
        success = self.score < self.detector.threshold and self.intent >= self.tau
        reward = compute_reward(prev_logit, self.logit, self.intent, changed, success,
                                self.alpha, self.beta, self.gamma, self.tau, self.success_bonus)
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
