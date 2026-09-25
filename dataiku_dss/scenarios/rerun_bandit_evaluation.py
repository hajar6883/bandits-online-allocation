"""Scenario: rebuild the whole Flow and log the ranking."""

import dataiku
from dataiku.scenario import Scenario

scenario = Scenario()

scenario.build_dataset(
    "bandit_ope_results",
    build_mode="RECURSIVE_FORCED_BUILD",
    project_key=dataiku.default_project_key(),
)

results = dataiku.Dataset("bandit_ope_results").get_dataframe()
results = results.sort_values("mean_reward", ascending=False)

lines = ["Bandit off-policy evaluation, mean reward per policy", ""]
for _, row in results.iterrows():
    lines.append(
        "  %-20s %.4f  (%s, %d arms, %d rounds x %d trials)"
        % (
            row["policy"],
            row["mean_reward"],
            row["family"],
            row["n_arms"],
            row["n_rounds"],
            row["n_trials"],
        )
    )

report = "\n".join(lines)
print(report)

best = results.iloc[0]
scenario.set_scenario_variables(
    best_policy=str(best["policy"]),
    best_mean_reward=float(best["mean_reward"]),
    bandit_run_id=str(best["run_id"]),
    bandit_report=report,
)
