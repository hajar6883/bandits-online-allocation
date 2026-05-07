from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from .bandits.base import Bandit, ContextualBandit


def run_non_contextual(
    learner: Bandit,
    time_horizon: int,
    n_trials: int,
    reward_dict: dict[int, Sequence[float]],
) -> np.ndarray:
    all_rewards = []
    for _ in range(n_trials):
        learner.clear()
        rewards = []
        for _t in range(time_horizon):
            arm = learner.chooseArmToPlay()
            reward = random.choice(reward_dict[arm])
            learner.receiveReward(arm, reward)
            rewards.append(reward)
        all_rewards.append(rewards)
    return np.array(all_rewards)


def run_contextual(
    learner: ContextualBandit,
    time_horizon: int,
    n_trials: int,
    reward_dict: dict[int, Sequence[float]],
    context_dict: dict[int, np.ndarray],
) -> np.ndarray:
    all_rewards = []
    for _ in range(n_trials):
        learner.clear()
        rewards = []
        for _t in range(time_horizon):
            arm, ctx_idx = learner.chooseArmToPlay(context_dict)
            reward = reward_dict[arm][ctx_idx]
            context = context_dict[arm][ctx_idx]
            learner.receiveReward(arm, reward, context)
            rewards.append(reward)
        all_rewards.append(rewards)
    return np.array(all_rewards)


def cumulative_average(rewards: np.ndarray) -> np.ndarray:
    mean_per_round = np.mean(rewards, axis=0)
    cum = np.cumsum(mean_per_round)
    rounds = np.arange(1, len(cum) + 1)
    return cum / rounds


def top_k_arms(scores: np.ndarray, k: int = 5) -> np.ndarray:
    return np.argpartition(scores, -k)[-k:]
