from __future__ import annotations

import numpy as np

from .base import Bandit


class UCB(Bandit):
    def __init__(self, nbArms: int, alpha: float = 0.5):
        self.nbArms = nbArms
        self.alpha = alpha
        self.clear()

    def clear(self) -> None:
        self.N_a = np.zeros(self.nbArms)
        self.ucb = np.full(self.nbArms, np.inf)
        self.t = 0
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self) -> int:
        self.t += 1
        return int(np.argmax(self.ucb))

    def receiveReward(self, arm: int, reward: float) -> None:
        self.cumRewards[arm] += reward
        self.N_a[arm] += 1
        Q_a = self.cumRewards[arm] / self.N_a[arm]
        self.ucb[arm] = Q_a + np.sqrt((self.alpha * np.log(self.t)) / self.N_a[arm])

    def name(self) -> str:
        return "UCB"
