from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bandits import (
    EpsilonGreedy,
    LinearThompsonSampling,
    LinUCB,
    ThompsonSampling,
    UCB,
)
from src.data_loader import prepare
from src.evaluation import run_contextual, run_non_contextual
from src.plots import cumulative_reward_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--ucb-alpha", type=float, default=0.5)
    parser.add_argument("--linucb-alpha", type=float, default=0.3)
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, reward_dict, context_dict = prepare(args.data)
    n_arms = len(reward_dict)
    n_features = next(iter(context_dict.values())).shape[1]
    print(f"Loaded {n_arms} arms, context dim = {n_features}.")

    runs = {}

    for label, learner in [
        ("Epsilon Greedy", EpsilonGreedy(n_arms, args.epsilon)),
        ("UCB", UCB(n_arms, args.ucb_alpha)),
        ("Thompson Sampling", ThompsonSampling(n_arms)),
    ]:
        rewards = run_non_contextual(learner, args.rounds, args.trials, reward_dict)
        runs[label] = rewards
        print(f"{label} average reward: {np.mean(rewards):.4f}")

    for label, learner in [
        ("LinUCB", LinUCB(n_arms, n_features, args.linucb_alpha)),
        ("LinTS", LinearThompsonSampling(n_arms, n_features)),
    ]:
        rewards = run_contextual(learner, args.rounds, args.trials, reward_dict, context_dict)
        runs[label] = rewards
        print(f"{label} average reward: {np.mean(rewards):.4f}")

    cumulative_reward_curves(runs, save_to=results_dir / "cumulative_reward_all.png")


if __name__ == "__main__":
    main()
