"""Bandit policies and the off-policy evaluation harness."""

from .base import Bandit, ContextualBandit, break_tie
from .epsilon_greedy import EpsilonGreedy
from .ucb import UCB
from .thompson import ThompsonSampling
from .linucb import LinUCB
from .lin_thompson import LinearThompsonSampling
from .evaluation import (
    cumulative_average,
    run_contextual,
    run_non_contextual,
    top_k_arms,
)

__all__ = [
    "Bandit",
    "ContextualBandit",
    "break_tie",
    "EpsilonGreedy",
    "UCB",
    "ThompsonSampling",
    "LinUCB",
    "LinearThompsonSampling",
    "run_non_contextual",
    "run_contextual",
    "cumulative_average",
    "top_k_arms",
]
