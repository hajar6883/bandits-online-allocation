from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bandits import LinUCB, LinearThompsonSampling
from src.data_loader import prepare
from src.evaluation import run_contextual
from src.plots import arm_frequency, cumulative_reward_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, reward_dict, context_dict = prepare(args.data)
    n_arms = len(reward_dict)
    n_features = next(iter(context_dict.values())).shape[1]
    print(f"Loaded {n_arms} arms, context dim = {n_features}.")

    learners = {
        "LinUCB": LinUCB(n_arms, n_features, alpha=args.alpha),
        "LinTS": LinearThompsonSampling(n_arms, n_features),
    }

    runs = {}
    for label, learner in learners.items():
        rewards = run_contextual(
            learner, args.rounds, args.trials, reward_dict, context_dict
        )
        runs[label] = rewards
        print(f"{label} average reward: {np.mean(rewards):.4f}")
        arm_frequency(
            learner.N_a,
            title=f"Frequency of Chosen Arms - {label}",
            save_to=results_dir / f"freq_{label.lower()}.png",
        )

    cumulative_reward_curves(runs, save_to=results_dir / "cumulative_reward_contextual.png")


if __name__ == "__main__":
    main()
