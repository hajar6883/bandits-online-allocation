from __future__ import annotations

import numpy as np

from .base import Bandit


class ThompsonSampling(Bandit):
    def __init__(self, nbArms: int, a: float = 1.0, b: float = 1.0):
        self.nbArms = nbArms
        self.a = a
        self.b = b
        self.clear()

    def clear(self) -> None:
        self.N_a = np.zeros(self.nbArms)
        self.pi = np.full(self.nbArms, np.inf)
        self.t = 0
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self) -> int:
        self.t += 1
        return int(np.argmax(self.pi))

    def receiveReward(self, arm: int, reward: float) -> None:
        S_a = self.cumRewards[arm]
        self.pi[arm] = np.random.beta(
            self.a + S_a, max(self.b + self.N_a[arm] - S_a, 0.1)
        )
        self.cumRewards[arm] += reward
        self.N_a[arm] += 1

    def name(self) -> str:
        return "Thompson Sampling"
