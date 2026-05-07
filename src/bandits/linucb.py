from __future__ import annotations

import numpy as np

from .base import ContextualBandit, break_tie


class LinUCB(ContextualBandit):
    def __init__(self, n_arms: int, n_dims: int, alpha: float):
        self.n_arms = n_arms
        self.n_features = n_dims
        self.alpha = alpha
        self.clear()

    def clear(self) -> None:
        self.N_a = np.zeros(self.n_arms)
        self.post_dist = np.zeros(self.n_arms)
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.inv_A = [np.linalg.inv(self.A[i]) for i in range(self.n_arms)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.theta = [np.zeros(self.n_features) for _ in range(self.n_arms)]

    def chooseArmToPlay(self, context_dict: dict) -> tuple[int, int]:
        context_index_chosen_by_arm = np.zeros(self.n_arms, dtype=int)
        for arm in range(self.n_arms):
            ctx_idx = 0
            context_index_chosen_by_arm[arm] = ctx_idx
            x = np.array(context_dict[arm][ctx_idx])
            self.post_dist[arm] = (
                self.theta[arm] @ x
                + self.alpha * np.sqrt(x.T @ self.inv_A[arm] @ x)
            )
        arm = break_tie(self.post_dist)
        return int(arm), int(context_index_chosen_by_arm[arm])

    def receiveReward(self, arm: int, reward: float, context: np.ndarray) -> None:
        self.N_a[arm] += 1
        x = np.asarray(context).reshape(-1, 1)
        self.A[arm] = self.A[arm] + x @ x.T
        self.inv_A[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] = self.b[arm] + reward * np.asarray(context)
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def name(self) -> str:
        return "LinUCB"
