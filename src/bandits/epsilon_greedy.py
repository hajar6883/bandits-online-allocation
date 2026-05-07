from __future__ import annotations

import numpy as np

from .base import Bandit


class EpsilonGreedy(Bandit):
    def __init__(self, nbArms: int, epsilon: float = 0.05):
        self.nbArms = nbArms
        self.epsilon = epsilon
        self.clear()

    def clear(self) -> None:
        self.N_a = np.zeros(self.nbArms)
        self.Q = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.nbArms))
        return int(np.argmax(self.Q))

    def receiveReward(self, arm: int, reward: float) -> None:
        self.cumRewards[arm] += reward
        self.N_a[arm] += 1
        self.Q[arm] = self.cumRewards[arm] / self.N_a[arm]

    def name(self) -> str:
        return "Epsilon Greedy"
