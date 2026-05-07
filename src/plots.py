from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import cumulative_average


def arm_frequency(counts: np.ndarray, title: str = "Frequency of Chosen Arms",
                  save_to: str | Path | None = None) -> None:
    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.xlabel("Arms (Videos)")
    plt.ylabel("Frequency")
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()


def cumulative_reward_curves(
    runs: Mapping[str, np.ndarray],
    save_to: str | Path | None = None,
) -> None:
    plt.figure()
    for label, rewards in runs.items():
        avg = cumulative_average(rewards)
        plt.plot(np.arange(1, len(avg) + 1), avg, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Cumulative Average Reward over Time")
    plt.legend()
    plt.grid(True)
    if save_to is not None:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()


def predictions_over_time(
    correct: list[int],
    missed: list[int],
    save_to: str | Path | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(correct)), correct, label="Correct Predictions")
    plt.plot(range(len(missed)), missed, label="Missed Opportunities")
    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.title("Evolution of Correct Predictions and Missed Opportunities")
    plt.legend()
    plt.grid(True)
    if save_to is not None:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()
