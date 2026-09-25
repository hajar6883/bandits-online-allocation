"""Step 3: run the five bandit policies off-policy and write the results."""

import json
import random
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import dataiku

from bandit_lib import (
    EpsilonGreedy,
    LinearThompsonSampling,
    LinUCB,
    ThompsonSampling,
    UCB,
    cumulative_average,
    run_contextual,
    run_non_contextual,
    top_k_arms,
)
from kuairec_prep import load_arm_dicts

variables = dataiku.get_custom_variables()
n_rounds = int(variables.get("n_rounds", 500))
n_trials = int(variables.get("n_trials", 250))
epsilon = float(variables.get("epsilon", 0.1))
ucb_alpha = float(variables.get("ucb_alpha", 0.5))
linucb_alpha = float(variables.get("linucb_alpha", 0.3))

raw_seed = str(variables.get("random_seed", "")).strip()
random_seed = int(raw_seed) if raw_seed not in ("", "None", "null") else None
if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    print("seeded with %d" % random_seed)

run_id = str(uuid.uuid4())
run_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

context_df = dataiku.Dataset("kuairec_bandit_context").get_dataframe()
reward_dict, context_dict, feature_columns = load_arm_dicts(context_df)

n_arms = len(reward_dict)
n_features = len(feature_columns)
arm_to_video = (
    context_df.drop_duplicates(subset=["arm_id"])
    .set_index("arm_id")["video_id"]
    .to_dict()
)
print("Loaded %d arms, context dim = %d." % (n_arms, n_features))
print("horizon = %d rounds x %d trials" % (n_rounds, n_trials))

non_contextual = [
    ("Epsilon Greedy", EpsilonGreedy(n_arms, epsilon), {"epsilon": epsilon}),
    ("UCB", UCB(n_arms, ucb_alpha), {"alpha": ucb_alpha}),
    ("Thompson Sampling", ThompsonSampling(n_arms), {"a": 1.0, "b": 1.0}),
]
contextual = [
    ("LinUCB", LinUCB(n_arms, n_features, linucb_alpha), {"alpha": linucb_alpha}),
    ("LinTS", LinearThompsonSampling(n_arms, n_features), {"alpha": 1.0}),
]

result_rows = []
curve_rows = []
frequency_rows = []

TOP_K_ATTR = {
    "Epsilon Greedy": "Q",
    "UCB": "ucb",
    "Thompson Sampling": "pi",
}


def record(label, family, learner, rewards, params):
    curve = cumulative_average(rewards)

    top_arms = None
    attr = TOP_K_ATTR.get(label)
    if attr is not None:
        top_arms = sorted(int(a) for a in top_k_arms(getattr(learner, attr), 5))

    result_rows.append({
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "policy": label,
        "family": family,
        "n_arms": n_arms,
        "n_features": n_features if family == "contextual" else None,
        "n_rounds": n_rounds,
        "n_trials": n_trials,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "final_cumulative_avg_reward": float(curve[-1]),
        "total_reward_per_trial": float(np.sum(rewards) / n_trials),
        "params": json.dumps(params, sort_keys=True),
        "top_5_arms": json.dumps(top_arms) if top_arms is not None else None,
    })

    mean_per_round = np.mean(rewards, axis=0)
    for t in range(len(curve)):
        curve_rows.append({
            "run_id": run_id,
            "policy": label,
            "family": family,
            "round": t + 1,
            "mean_reward_at_round": float(mean_per_round[t]),
            "cumulative_avg_reward": float(curve[t]),
        })

    for arm_id, pulls in enumerate(learner.N_a):
        if pulls > 0:
            frequency_rows.append({
                "run_id": run_id,
                "policy": label,
                "arm_id": int(arm_id),
                "video_id": int(arm_to_video[arm_id]),
                "n_pulls": float(pulls),
            })

    print("%s average reward: %.4f" % (label, np.mean(rewards)))


for label, learner, params in non_contextual:
    rewards = run_non_contextual(learner, n_rounds, n_trials, reward_dict)
    record(label, "non-contextual", learner, rewards, params)

for label, learner, params in contextual:
    rewards = run_contextual(learner, n_rounds, n_trials, reward_dict, context_dict)
    record(label, "contextual", learner, rewards, params)

results = pd.DataFrame(result_rows).sort_values("mean_reward", ascending=False)
curves = pd.DataFrame(curve_rows)
frequency = pd.DataFrame(frequency_rows)

print("\nRanking by mean reward:")
for _, row in results.iterrows():
    print("  %-20s %.4f" % (row["policy"], row["mean_reward"]))

dataiku.Dataset("bandit_ope_results").write_with_schema(results)
dataiku.Dataset("bandit_reward_curves").write_with_schema(curves)
dataiku.Dataset("bandit_arm_frequency").write_with_schema(frequency)
