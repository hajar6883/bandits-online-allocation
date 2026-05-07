from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bandits import EpsilonGreedy, UCB, ThompsonSampling
from src.data_loader import prepare
from src.evaluation import run_non_contextual, top_k_arms
from src.plots import arm_frequency, cumulative_reward_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--ucb-alpha", type=float, default=0.5)
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, reward_dict, _ = prepare(args.data)
    n_arms = len(reward_dict)
    print(f"Loaded {n_arms} arms.")

    learners = {
        "Epsilon Greedy": EpsilonGreedy(n_arms, epsilon=args.epsilon),
        "UCB": UCB(n_arms, alpha=args.ucb_alpha),
        "Thompson Sampling": ThompsonSampling(n_arms),
    }

    runs = {}
    for label, learner in learners.items():
        rewards = run_non_contextual(learner, args.rounds, args.trials, reward_dict)
        runs[label] = rewards
        print(f"{label} average reward: {np.mean(rewards):.4f}")
        arm_frequency(
            learner.N_a,
            title=f"Frequency of Chosen Arms - {label}",
            save_to=results_dir / f"freq_{label.replace(' ', '_').lower()}.png",
        )

    cumulative_reward_curves(runs, save_to=results_dir / "cumulative_reward.png")

    print("\nTop-5 arms by learner:")
    print("  Epsilon Greedy:", top_k_arms(learners["Epsilon Greedy"].Q, 5))
    print("  UCB           :", top_k_arms(learners["UCB"].ucb, 5))
    print("  Thompson      :", top_k_arms(learners["Thompson Sampling"].pi, 5))


if __name__ == "__main__":
    main()
