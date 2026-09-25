from __future__ import annotations

import random

import numpy as np

from .base import ContextualBandit


class LinearThompsonSampling(ContextualBandit):
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.clear()

    def clear(self) -> None:
        self.N_a = np.zeros(self.n_arms)
        self.V = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.n_features))

    def chooseArmToPlay(self, context_dict: dict) -> tuple[int, int]:
        context_index_chosen_by_arm = np.zeros(self.n_arms, dtype=int)
        p = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            inv_V = np.linalg.inv(self.V[arm])
            mu_hat = inv_V @ self.b[arm]
            covariance = inv_V * (mu_hat ** 2)
            sampled_theta = np.random.multivariate_normal(mu_hat, covariance)

            ctx_idx = random.randint(0, len(context_dict[arm]) - 1)
            context_index_chosen_by_arm[arm] = ctx_idx
            x = np.array(context_dict[arm][ctx_idx])
            p[arm] = sampled_theta @ x

        chosen_arm = int(np.argmax(p))
        return chosen_arm, int(context_index_chosen_by_arm[chosen_arm])

    def receiveReward(self, arm: int, reward: float, context: np.ndarray) -> None:
        self.N_a[arm] += 1
        x = np.asarray(context)
        self.V[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def name(self) -> str:
        return "LinearThompsonSampling"
