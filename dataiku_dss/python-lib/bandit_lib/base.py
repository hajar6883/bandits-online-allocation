from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Bandit(ABC):
    nbArms: int
    N_a: np.ndarray

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def chooseArmToPlay(self) -> int: ...

    @abstractmethod
    def receiveReward(self, arm: int, reward: float) -> None: ...

    @abstractmethod
    def name(self) -> str: ...


class ContextualBandit(ABC):
    n_arms: int
    n_features: int
    N_a: np.ndarray

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def chooseArmToPlay(self, context_dict: dict) -> tuple[int, int]: ...

    @abstractmethod
    def receiveReward(self, arm: int, reward: float, context: np.ndarray) -> None: ...

    @abstractmethod
    def name(self) -> str: ...


def break_tie(values: np.ndarray) -> int:
    indices = np.argwhere(values == np.max(values))
    return int(indices[np.random.randint(0, len(indices))][0])
