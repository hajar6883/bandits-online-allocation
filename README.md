# Multi-Armed Bandits for Personalized Recommendation

INF581 (Ecole Polytechnique) RL project. Contextual and non-contextual bandit
algorithms applied to the KuaiRec short-video recommendation dataset, with
off-policy evaluation against logged user/video/watch-ratio interactions.

## Project Structure

```
mab_bandits/
├── src/
│   ├── data_loader.py        KuaiRec loading, sub-sampling, reward / context dicts
│   ├── evaluation.py         off-policy evaluation harness, top-K helpers
│   ├── plots.py              arm frequency, cumulative reward curves
│   └── bandits/
│       ├── base.py           Bandit / ContextualBandit ABCs
│       ├── epsilon_greedy.py
│       ├── ucb.py
│       ├── thompson.py
│       ├── linucb.py
│       └── lin_thompson.py
├── scripts/
│   ├── run_non_contextual.py Part 1: epsilon-greedy / UCB / TS
│   ├── run_contextual.py     Part 2: LinUCB / LinTS
│   └── compare_all.py        all five learners on one plot
├── report/                   team report (PDF)
├── data/                     KuaiRec CSVs (not committed, see data/README.md)
└── results/                  generated plots
```

## Methodology

Watch-ratio is thresholded into a Bernoulli reward: `reward = 1` if the user
watched more than 80% of the video, 0 otherwise.

For off-policy evaluation, at each round the learner picks an arm (and, in the
contextual case, a context for that arm), then we fetch a logged reward for
that arm from the interaction data.

Algorithms:

| Family          | Algorithm                |
|-----------------|--------------------------|
| Non-contextual  | epsilon-greedy           |
|                 | UCB1                     |
|                 | Thompson Sampling (Beta) |
| Contextual      | LinUCB (disjoint)        |
|                 | Linear Thompson Sampling |

A separate simulation in the notebook uses the
[`mabwiser`](https://github.com/fidelity/mabwiser) library with
`LinGreedy + KNearest` and an experience-score-aware reward shaping that
distinguishes explorer vs exploiter users on the held-out split.

## Run

```bash
pip install -r requirements.txt

# put KuaiRec CSVs in data/ (see data/README.md)

python scripts/run_non_contextual.py --rounds 500 --trials 250
python scripts/run_contextual.py --rounds 500 --trials 250 --alpha 0.3
python scripts/compare_all.py
```

## Report

Full write-up with results and discussion in
`report/INF581_Team_Project_Report.pdf`.
