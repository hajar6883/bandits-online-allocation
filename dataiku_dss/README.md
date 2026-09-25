# KuaiRec Bandits on Dataiku DSS (Free Edition)

This folder is the Dataiku port of the INF581 multi-armed-bandit project. The
Python in `../src/` runs as three scripts against CSVs on disk; here the same
work is split into a DSS **Flow** of three Python recipes, refreshed by a
**scenario** and summarised on a **dashboard**.

**The evaluation logic and its results are unchanged.** The five policy classes
in `python-lib/bandit_lib/` are byte-identical copies of `src/bandits/*.py`, and
`bandit_lib/evaluation.py` is `src/evaluation.py` with one import line adjusted.
What changed is only where data comes from (`dataiku.Dataset` instead of
`pd.read_csv`) and where numbers go (output datasets instead of matplotlib
windows). See [Equivalence](#equivalence) for the checks.

---

## 1. What is in this folder

```
dataiku_dss/
├── python-lib/                     -> goes into the project's Libraries
│   ├── kuairec_prep.py             prep + feature helpers (from src/data_loader.py)
│   └── bandit_lib/
│       ├── __init__.py
│       ├── base.py                 |
│       ├── epsilon_greedy.py       | byte-identical copies of
│       ├── ucb.py                  | src/bandits/*.py
│       ├── thompson.py             |
│       ├── linucb.py               |
│       ├── lin_thompson.py         |
│       └── evaluation.py           src/evaluation.py (import path only)
├── recipes/                        -> paste into DSS Python recipes
│   ├── compute_kuairec_interactions_prepared.py   step 1: preparation
│   ├── compute_kuairec_bandit_context.py          step 2: features
│   └── compute_bandit_ope_results.py              step 3: evaluation
├── scenarios/
│   └── rerun_bandit_evaluation.py  step 4: scenario
└── project-variables.json          default project variables
```

Each recipe file is named `compute_<primary output dataset>.py`, the DSS
convention, so you can tell at a glance which recipe builds what.

## 2. The Flow

```
kuairec_small_matrix    ─┐
kuairec_user_features   ─┼─► [ py ] ─► kuairec_interactions_prepared
kuairec_item_categories ─┘   prepare

kuairec_interactions_prepared ─► [ py ] ─┬─► kuairec_bandit_context
                                features  ├─► kuairec_context_schema
                                          └─► kuairec_user_profile

kuairec_bandit_context ─► [ py ] ─┬─► bandit_ope_results
                           evaluate├─► bandit_reward_curves
                                   └─► bandit_arm_frequency
```

| Dataset | Rows (defaults) | What it holds |
|---|---|---|
| `kuairec_interactions_prepared` | ~10.8k x 61 | cleaned sub-sample, user features + 32 item tag one-hots |
| `kuairec_bandit_context` | ~10.8k x 30 | `arm_id, video_id, ctx_index, reward, f_0..f_25` |
| `kuairec_context_schema` | 26 | maps `f_i` back to its source column |
| `kuairec_user_profile` | ~135 | per-user `exp_score` (exploration score) |
| `bandit_ope_results` | 5 | one row per policy: mean reward, std, params |
| `bandit_reward_curves` | 5 x n_rounds | cumulative-average reward curve per policy |
| `bandit_arm_frequency` | 5 x arms pulled | pull counts per policy and arm |

The last two replace the PNGs in `results/`: `bandit_reward_curves` is the data
behind `cumulative_reward.png`, `bandit_arm_frequency` the data behind the
`freq_*.png` bar charts. As datasets they can be charted natively in DSS and
dropped on a dashboard.

---

## 3. Build it in the UI, step by step

### 3.0 Create the project and a code env

1. On the DSS home page: **+ New Project > Blank project**. Name it
   `KuaiRec Bandits`; DSS derives the key `KUAIREC_BANDITS`.
2. **No code env is needed on a stock DSS 15 macOS install.** Its built-in
   Python is 3.12 with numpy 2.0, pandas 2.2 and scikit-learn 1.6 already
   present, which satisfies `requirements.txt`. Verify with
   **Administration > Code envs** only if a recipe later fails on an import.

   If you do need one: **+ New Python env** named `py_kuairec_bandits`, and in
   **Packages to install** paste `numpy>=1.24`, `pandas>=2.0`,
   `scikit-learn>=1.3`. matplotlib/seaborn are not needed - DSS draws the
   charts. `mabwiser` is only used by the notebook-side simulation.

3. Only if you built a code env: **... (top right) > Settings > Code env**,
   select *Select an environment* and pick `py_kuairec_bandits`.

### 3.1 Install the project library

1. In the top navigation bar open the **`</>` (code) menu > Libraries**.
2. You land in the project's library editor, which already contains a
   `python/` folder.
3. Recreate this structure under `python/` (right-click a folder for
   *Create file* / *Create folder*) and paste each file's contents from
   `dataiku_dss/python-lib/`:

   ```
   python/
   ├── kuairec_prep.py
   └── bandit_lib/
       ├── __init__.py
       ├── base.py
       ├── epsilon_greedy.py
       ├── ucb.py
       ├── thompson.py
       ├── linucb.py
       ├── lin_thompson.py
       └── evaluation.py
   ```

4. **Save all**. Everything under `python/` is on the PYTHONPATH of every
   recipe, notebook and scenario in the project, which is why the recipes can
   just say `from bandit_lib import LinUCB`.

> **Much faster on a local install** (this is what was actually used): copy the
> files straight into the DSS data dir and reload the Libraries page. On the
> macOS Free Edition the data dir is `~/Library/DataScienceStudio/dss_home`:
>
> ```bash
> cp -r dataiku_dss/python-lib/* \
>   ~/Library/DataScienceStudio/dss_home/config/projects/KUAIREC_BANDITS/lib/python/
> ```
>
> DSS picks the files up without a restart - recipes get
> `lib/python` on `sys.path` at run time, so no cache invalidation is needed.

### 3.2 Set the project variables

1. **... (top right, the "More options" menu) > Variables** - in DSS 15 this
   is its own entry, *not* under Settings.
2. In **Global variables**, paste the contents of `project-variables.json`:

   ```json
   {
     "keep_video_frac": 0.025,
     "keep_user_frac": 0.1,
     "include_item_features": "false",
     "n_rounds": 500,
     "n_trials": 250,
     "epsilon": 0.1,
     "ucb_alpha": 0.5,
     "linucb_alpha": 0.3,
     "random_seed": ""
   }
   ```

3. **Save**. These are the CLI flags of the original scripts, promoted to
   project settings so the scenario and the recipes share them.

   | Variable | Original equivalent |
   |---|---|
   | `n_rounds`, `n_trials` | `--rounds`, `--trials` |
   | `epsilon` | `--epsilon` (`run_non_contextual.py`) |
   | `ucb_alpha` | `--ucb-alpha` |
   | `linucb_alpha` | `--linucb-alpha` / `--alpha` |
   | `keep_video_frac`, `keep_user_frac` | hard-coded in `build_subsample` |
   | `random_seed` | none - see [Equivalence](#equivalence) |
   | `include_item_features` | none - see [Equivalence](#equivalence) |

### 3.3 Create the three input datasets

`small_matrix.csv` is ~387 MB, well past what the upload widget is comfortable
with, so point DSS at the folder instead of uploading.

1. **Flow > + Dataset > Filesystem** (under the *Files* heading).
2. Set **Read from** to **`filesystem_root`** - the default `filesystem_folders`
   is rooted inside the DSS data dir and cannot see your project folder.
3. Put the **absolute** path in **Path**:
   `/Users/grace/Desktop/M2/mab_bandits/data/small_matrix.csv`
   (adjust if DSS runs elsewhere - it must be readable by the DSS process).
4. Click **TEST**. It should report *Used /small_matrix.csv (387.34 MB) to parse
   data* and *found 8 columns*. On a 387 MB file this takes ~15 s.
5. Name it **`kuairec_small_matrix`** and **CREATE**.
6. Repeat for `/user_features.csv` -> **`kuairec_user_features`** (31 columns)
   and `/item_categories.csv` -> **`kuairec_item_categories`** (2 columns).

#### 3.3b Fix the column types - do not skip this

DSS creates a CSV filesystem dataset with **every column typed `string`**. That
silently breaks the port, in two ways that do not raise an error:

* `build_subsample` does `sort_values("video_id")`. On strings that sorts
  lexicographically - `"10"` before `"9"` - so the 2.5% head takes a completely
  different set of rows than the original pandas code, and every downstream
  number changes.
* `watch_ratio > 0.8` compares a string to a float, and the merge keys on both
  sides must agree in type or the join silently produces no matches.

For **each** of the three datasets:

1. Open it, then **Settings > Schema**.
2. Click **CHECK NOW**. It must say *Schema and data are consistent*; this also
   enables the next button.
3. Click **INFER TYPES FROM DATA** and **CONFIRM**. DSS warns that it samples
   only the head of the file - that is fine here, the KuaiRec columns are
   uniformly typed.
4. **SAVE**.

Then two manual corrections, because inference guesses differently from pandas:

| Dataset | Column | Inferred | Set it to | Why |
|---|---|---|---|---|
| `kuairec_small_matrix` | `time` | `datetime no tz` | **string** | `pd.read_csv` leaves it a string. As a datetime, any unparseable value becomes NaT and is then dropped by the `dropna()` that runs *before* the column is discarded - changing the row set. |
| `kuairec_item_categories` | `feat` | `array<...>` | **string** | It holds a stringified list like `[8, 27]`, which `parse_item_categories` evaluates. (The library handles either form, but string is what the original code saw.) |

After this the schemas should read:

* `kuairec_small_matrix` - `user_id`, `video_id`, `play_duration`,
  `video_duration` bigint; `time` string; `date`, `timestamp`, `watch_ratio` double
* `kuairec_user_features` - `user_id` + 25 numeric columns bigint;
  `user_active_degree` and the four `*_range` columns string
* `kuairec_item_categories` - `video_id` bigint, `feat` string

### 3.4 Recipe 1 - data preparation

1. In the Flow, select **`kuairec_small_matrix`**, then **Actions (right
   panel) > Python recipe**.
2. **Inputs**: add `kuairec_user_features` and `kuairec_item_categories`
   alongside `kuairec_small_matrix`.
3. **Outputs**: **+ Add** > new dataset named
   **`kuairec_interactions_prepared`**, store into the project's default
   connection (`filesystem_managed`), **Create dataset**.
4. **Create recipe**. Replace the generated code with the contents of
   `recipes/compute_kuairec_interactions_prepared.py`.
5. **Run**. The log should print the sub-sample size
   (`after sub-sampling: 10782 rows, 85 videos, 135 users` with default
   fractions on the full KuaiRec small matrix).

What it does: inner-joins interactions to `user_features`, drops incomplete
rows, keeps 2.5% of the video-sorted frame then 10% of the user-sorted one,
drops the leakage columns (`play_duration`, `video_duration`, the timestamps)
and the redundant `*_range` columns, label-encodes `user_active_degree`, and
finally left-joins the 32 item tag one-hots built from `item_categories`.

The item join is deliberately **after** the sub-sample and is a **left** join,
so it cannot change which rows the evaluation sees. The recipe raises if the
row count moves.

### 3.5 Recipe 2 - context features

1. Select **`kuairec_interactions_prepared`** > **Python recipe**.
2. **Outputs**: create three datasets - **`kuairec_bandit_context`**,
   **`kuairec_context_schema`**, **`kuairec_user_profile`**.
3. **Create recipe**, paste `recipes/compute_kuairec_bandit_context.py`, **Run**.

What it does: thresholds `watch_ratio > 0.8` into the Bernoulli reward, groups
the interactions by `video_id` (each video is an arm, enumerated `0..84`), and
flattens the per-arm reward list and context matrix into one long dataset keyed
by `(arm_id, ctx_index)`. That pair is the Flow-friendly form of the original
in-memory `reward_dict` / `context_dict`, and keeping `ctx_index` explicit is
what guarantees `reward_dict[a][i]` still lines up with `context_dict[a][i]`
after a round-trip through storage.

The context vector is the user-feature block - 26 columns,
`user_active_degree` through `onehot_feat17` - which is exactly what the
original `build_context_dict` selected. `kuairec_context_schema` spells out the
`f_i` -> column-name mapping so the numbers stay readable in the UI.

It also writes `kuairec_user_profile`, the per-user exploration score from
`exp_score_generator` (share of distinct tags among the videos a user liked,
min-max scaled). It feeds the dashboard and is not used by the five policies.

### 3.6 Recipe 3 - the five policies, off-policy

1. Select **`kuairec_bandit_context`** > **Python recipe**.
2. **Outputs**: create **`bandit_ope_results`**, **`bandit_reward_curves`**,
   **`bandit_arm_frequency`**.
3. **Create recipe**, paste `recipes/compute_bandit_ope_results.py`, **Run**.

This is `scripts/compare_all.py` with dataset I/O. It rebuilds the two
dictionaries, then runs the untouched evaluation harness for epsilon-greedy,
UCB1, Thompson Sampling, LinUCB and LinTS: each round the learner picks an arm
(and, contextually, a context index for it) and a logged reward for that arm is
read back from the interaction data. No counterfactual model, same as before.

At the defaults (500 rounds x 250 trials x 85 arms) the contextual policies
invert an 26x26 matrix per arm per round, so expect several minutes. Drop
`n_trials` to 25 while you are wiring the Flow up, then put it back.

The log prints the same per-policy lines the original script did:

```
Loaded 85 arms, context dim = 26.
Epsilon Greedy average reward: 0.51xx
...
```

### 3.7 Scenario - rerun the evaluation

**Point-and-click version** (the one to prefer on Free Edition):

1. **Scenarios > + New Scenario > Sequence of steps**, name it
   `Rerun bandit evaluation`.
2. **Steps > ADD STEP > Build / Train**.
3. **+ ADD ITEM > Dataset > `bandit_ope_results` > ADD**.
4. Leave **Build mode** as *Build upstream*, and set **Handling of
   dependencies** to **Force-build** ("Rebuild all datasets leading to the
   selected one"). That one step reruns preparation, features and evaluation
   in order. Then **SAVE**.
5. *(optional)* **+ Add Step > Send message** to get the ranking mailed to you.
6. **Settings > Triggers > + Add Trigger > Time-based trigger** if you want it
   nightly; leave it off for a manual rerun. Auto-triggers only fire while the
   DSS instance is running - Free Edition has no automation node.
7. **Save**, then **Run** to test.

**Scripted version**, if you want the ranking in the scenario log and in
scenario variables: **+ New Scenario > Custom Python**, then paste
`scenarios/rerun_bandit_evaluation.py` into the **Script** tab. It does the
same recursive rebuild, then reads `bandit_ope_results` back and sets
`best_policy`, `best_mean_reward`, `bandit_run_id` and `bandit_report` as
scenario variables (usable as `${best_policy}` in a later mail step).

Because every run stamps a fresh `run_id` and `run_timestamp`, you can switch
the three output datasets to **append** mode (dataset **Settings > Append
instead of overwrite**) and keep a history of runs to compare across scenario
executions.

### 3.8 Dashboard - mean reward per policy

Build each chart on its dataset's **Charts** tab by dragging column names from
the left rail into the **Show** (Y) and **By** (X / And) wells, then publish it.

**Tile 1 - mean reward per policy (the headline comparison)**

1. Open **`bandit_ope_results` > Charts**. Type is **Vertical bars** by default.
2. Drag **`mean_reward`** into **Show / Y** (it defaults to AVG - with one row
   per policy that is just the value).
3. Drag **`policy`** into **By / X**.
4. Drag **`family`** into the **And** well. The bars now colour by contextual
   vs non-contextual, which is the whole point of the comparison.
5. **PUBLISH > CREATE** to put it on the project's default dashboard.

**Tile 2 - cumulative average reward over time** (the `cumulative_reward.png` curve)

1. Open **`bandit_reward_curves` > Charts**, change the type dropdown from
   *Vertical bars* to **Lines**.
2. Drag **`cumulative_avg_reward`** into **Show / Y**.
3. Drag **`round`** into **By / X**. DSS bins it into 30 bins by default, which
   is fine here - the curves are smooth.
4. Drag **`policy`** into the **And** well to get one line per learner.
5. **PUBLISH > CREATE**.

**Tile 3 - arm pull frequency** (the `freq_*.png` bars)

1. Open **`bandit_arm_frequency` > Charts**, **Vertical bars**.
2. Drag **`n_pulls`** into **Show / Y**, **`arm_id`** into **By / X**,
   **`policy`** into **And**.
3. On the `arm_id` chip, open its **⋮ menu > Grouping > None, use raw values**.
   Leaving the default 10 bins *averages* pull counts inside each bin, which
   hides the whole point: each policy spikes hard on a handful of arms.
4. **PUBLISH > CREATE**.

**Tile 4 - run metadata**

On the dashboard, **+ NEW TILE > Dataset table** on `bandit_ope_results`,
showing `policy, family, mean_reward, std_reward, n_rounds, n_trials, params`.
It makes the numbers behind the charts auditable, and `run_timestamp` shows
when they were last refreshed.

Finally, **+ NEW TILE > Scenario** pointing at `Rerun bandit evaluation`, so the
dashboard carries its own rerun button.

---

## 3.9 Results from the built Flow

Produced by the `Rerun bandit evaluation` scenario at the defaults
(500 rounds x 250 trials, 85 arms, 26-dim context, unseeded):

| Policy | Family | Mean reward | Std | Total reward / trial |
|---|---|---|:--:|:--:|
| **LinUCB** | contextual | **0.9180** | 0.2744 | 459.0 |
| Epsilon Greedy | non-contextual | 0.7783 | 0.4154 | 389.1 |
| LinTS | contextual | 0.7533 | 0.4311 | 376.6 |
| Thompson Sampling | non-contextual | 0.7520 | 0.4319 | 376.0 |
| UCB | non-contextual | 0.6784 | 0.4671 | 339.2 |

LinUCB wins clearly, reproducing the original project's finding that the
disjoint contextual learner beats every non-contextual baseline. Note that
LinUCB's advantage is partly structural: its `chooseArmToPlay` always evaluates
context index 0 for each arm, so it exploits one fixed context per arm rather
than sampling one, as LinTS does. That is the original implementation, kept
unchanged.

**Runtime.** On the reference machine (Apple Silicon, 8 GB RAM) the full
scenario took **43 minutes**, almost all of it in LinTS: it draws a
26-dimensional multivariate normal for all 85 arms at every one of
500 x 250 rounds. Recipes 1 and 2 take ~30 s and ~10 s. Set `n_trials` to 25
while wiring the Flow up - it finishes in under two minutes and gives the same
ranking - then restore 250 for the final run.

**One pre-existing warning.** LinTS emits
`RuntimeWarning: covariance is not symmetric positive-semidefinite` from
`lin_thompson.py:30`. That comes from the original implementation, where the
sampling covariance is built as `inv_V * (mu_hat ** 2)` rather than a true
posterior covariance. It was left untouched because the brief was to keep the
evaluation logic unchanged, but it is worth revisiting if the work continues.

---

## 4. Equivalence

Two checks were run against the real KuaiRec CSVs in `../data/`:

1. **Same data structures.** Feeding one sub-sample through the original
   `build_reward_dict` / `build_context_dict` and through the port's
   `join_item_features -> build_context_frame -> load_arm_dicts` round-trip
   gives 85 arms and a 26-dim context on both sides, with **0 arms differing**
   in rewards and **0 differing** in contexts. The port's named-column
   selection is also asserted equal to the original's positional
   `df1.iloc[:, 1:]`.
2. **Same evaluation.** With a shared seed and the same dictionaries, the
   `(n_trials, n_rounds)` reward matrix from `src.evaluation` and from
   `bandit_lib.evaluation` is **element-wise identical** for all five policies.

Three deliberate departures, all opt-in and all off by default:

- **`random_seed`** is new. The original scripts never seeded, so leaving the
  variable empty reproduces the original unseeded behaviour exactly; set it
  only when you want a reproducible rerun. Because of that, two runs at the
  defaults will differ in the 3rd decimal - as they did before.
- **`include_item_features`** defaults to `"false"`. The 32 item tag one-hots
  are carried in `kuairec_interactions_prepared` (they are what
  `exp_score_generator` needs) but are kept **out** of the context vector,
  since widening the context would change the LinUCB / LinTS numbers. Set it to
  `"true"` to experiment; the original numbers no longer apply if you do.
- **Column selection by name.** `build_context_frame` selects the context
  columns by name where the original used `groupby(...).apply(lambda x:
  x.iloc[:, 1:])`. Same columns in the same order, but immune to the pandas
  2.2+ `groupby.apply(include_groups=...)` deprecation that will otherwise
  silently drop a column on pandas 3.

One caveat inherited from the original: `build_subsample` sorts with pandas'
default quicksort, which is not stable, so rows at the 2.5% / 10% cut points
can vary between builds of `kuairec_interactions_prepared`. This was already
true of `scripts/*.py` and was left as is.

## 5. Free Edition notes

Verified end to end on **DSS 15.0.2 (macOS, Apple Silicon)**, Free Edition,
against the real KuaiRec CSVs. Built-in Python 3.12.12 with numpy 2.0.2,
pandas 2.2.3, scikit-learn 1.6.1 - no code env was needed.


- Everything here fits Free Edition: Python recipes, project libraries, code
  envs, scenarios and dashboards are all included.
- The Free Edition licence has no expiry; it is generated from the form on
  first launch and never needs renewing. (The 14-day limit belongs to Dataiku
  Cloud, a different product.)
- No automation node, so scenario triggers only fire while your DSS instance is
  running. Manual runs and the dashboard scenario tile work regardless.
- Free Edition is single-user, so the dashboard is for your own review; there
  is no sharing with a reader group.
- Datasets are stored on the local filesystem connection. `write_with_schema`
  overwrites by default; switch a dataset to append mode if you want run
  history.

## 6. Mapping back to the original project

| Original | DSS |
|---|---|
| `src/data_loader.py: build_subsample` | recipe 1 via `kuairec_prep.build_subsample` |
| `src/data_loader.py: build_one_hot_items` | recipe 1 via `kuairec_prep.join_item_features` |
| `src/data_loader.py: build_reward_dict` / `build_context_dict` | recipe 2 via `build_context_frame` (+ `load_arm_dicts` in recipe 3) |
| `src/data_loader.py: exp_score_generator` | recipe 2 -> `kuairec_user_profile` |
| `src/bandits/*.py` | `python-lib/bandit_lib/` (unchanged) |
| `src/evaluation.py` | `python-lib/bandit_lib/evaluation.py` (unchanged) |
| `scripts/run_non_contextual.py` | recipe 3, non-contextual policies |
| `scripts/run_contextual.py` | recipe 3, contextual policies |
| `scripts/compare_all.py` | recipe 3 (all five) + scenario |
| `src/plots.py: cumulative_reward_curves` | `bandit_reward_curves` -> dashboard tile 2 |
| `src/plots.py: arm_frequency` | `bandit_arm_frequency` -> dashboard tile 3 |
| CLI flags | project variables |
