"""KuaiRec preparation and feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

WATCH_THRESHOLD = 0.8

DROP_COLUMNS = [
    "play_duration",
    "video_duration",
    "time",
    "date",
    "timestamp",
    "follow_user_num_range",
    "friend_user_num_range",
    "fans_user_num_range",
    "register_days_range",
]

NON_CONTEXT_COLUMNS = ["user_id", "video_id", "watch_ratio"]

N_TAGS = 32
TAG_COLUMNS = [f"feat_video{i}" for i in range(N_TAGS)]


def build_subsample(
    small_matrix: pd.DataFrame,
    user_features: pd.DataFrame,
    *,
    keep_video_frac: float = 0.025,
    keep_user_frac: float = 0.1,
) -> pd.DataFrame:
    df = pd.merge(small_matrix, user_features, on="user_id", how="inner").dropna()

    df = df.sort_values(by="video_id")
    df = df.iloc[: int(keep_video_frac * len(df))]

    df = df.sort_values(by="user_id")
    df = df.iloc[: int(keep_user_frac * len(df))]

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    label_encoder = LabelEncoder()
    df["user_active_degree"] = label_encoder.fit_transform(df["user_active_degree"])
    return df


def parse_item_categories(item_categories: pd.DataFrame) -> pd.DataFrame:
    out = item_categories.copy()
    if out["feat"].dtype == object and not isinstance(out["feat"].iloc[0], list):
        out["feat"] = out["feat"].map(eval)
    return out


def build_one_hot_items(
    item_categories: pd.DataFrame, n_tags: int = N_TAGS
) -> pd.DataFrame:
    item_categories = parse_item_categories(item_categories)
    rows = np.zeros((len(item_categories), n_tags), dtype=int)
    for i, feats in enumerate(item_categories["feat"].values):
        for f in feats:
            if 0 <= int(f) < n_tags:
                rows[i, int(f)] = 1
    one_hot = pd.DataFrame(rows, columns=[f"feat_video{i}" for i in range(n_tags)])
    return pd.concat(
        [item_categories[["video_id"]].reset_index(drop=True), one_hot], axis=1
    )


def join_item_features(
    interactions: pd.DataFrame,
    item_categories: pd.DataFrame,
    *,
    n_tags: int = N_TAGS,
) -> pd.DataFrame:
    one_hot = build_one_hot_items(item_categories, n_tags=n_tags)
    one_hot = one_hot.drop_duplicates(subset=["video_id"], keep="first")

    n_before = len(interactions)
    joined = interactions.merge(one_hot, on="video_id", how="left")
    if len(joined) != n_before:
        raise ValueError(
            f"item join changed the row count ({n_before} -> {len(joined)}); "
            "item_categories must have one row per video_id"
        )

    tag_columns = [f"feat_video{i}" for i in range(n_tags)]
    joined[tag_columns] = joined[tag_columns].fillna(0).astype(int)
    return joined


def watch_ratio_to_reward(watch_ratios) -> list[int]:
    return [1 if w > WATCH_THRESHOLD else 0 for w in watch_ratios]


def context_feature_columns(
    df: pd.DataFrame, *, include_item_features: bool = False
) -> list[str]:
    excluded = set(NON_CONTEXT_COLUMNS)
    if not include_item_features:
        excluded.update(TAG_COLUMNS)
    return [c for c in df.columns if c not in excluded]


def build_context_frame(
    df: pd.DataFrame, *, include_item_features: bool = False
) -> pd.DataFrame:
    feature_columns = context_feature_columns(
        df, include_item_features=include_item_features
    )

    frames = []
    for arm_id, (video_id, group) in enumerate(df.groupby("video_id", sort=True)):
        block = group[feature_columns].reset_index(drop=True)
        block.columns = [f"f_{i}" for i in range(len(feature_columns))]
        block.insert(0, "arm_id", arm_id)
        block.insert(1, "video_id", video_id)
        block.insert(2, "ctx_index", np.arange(len(group)))
        block.insert(3, "reward", watch_ratio_to_reward(group["watch_ratio"].tolist()))
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["arm_id", "ctx_index"]).reset_index(drop=True)


def context_schema(
    df: pd.DataFrame, *, include_item_features: bool = False
) -> pd.DataFrame:
    feature_columns = context_feature_columns(
        df, include_item_features=include_item_features
    )
    return pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_columns)),
            "feature_name": [f"f_{i}" for i in range(len(feature_columns))],
            "source_column": feature_columns,
        }
    )


def exp_score_generator(df: pd.DataFrame, n_tags: int = N_TAGS) -> pd.DataFrame:
    liked = df[df["watch_ratio"] >= 1]
    tag_columns = [f"feat_video{i}" for i in range(n_tags)]

    user_score = liked.groupby("user_id")[tag_columns].sum()
    user_score["exp_score"] = user_score.astype(bool).sum(axis=1) / n_tags

    scaler = MinMaxScaler()
    user_score["exp_score"] = scaler.fit_transform(user_score[["exp_score"]].values)

    return user_score[["exp_score"]].reset_index()


def load_arm_dicts(
    context_df: pd.DataFrame,
) -> tuple[dict[int, list[int]], dict[int, np.ndarray], list[str]]:
    feature_columns = sorted(
        (c for c in context_df.columns if c.startswith("f_")),
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not feature_columns:
        raise ValueError("no f_* context columns found in the context dataset")

    ordered = context_df.sort_values(["arm_id", "ctx_index"])

    reward_dict: dict[int, list[int]] = {}
    context_dict: dict[int, np.ndarray] = {}
    for arm_id, group in ordered.groupby("arm_id", sort=True):
        reward_dict[int(arm_id)] = group["reward"].astype(int).tolist()
        context_dict[int(arm_id)] = group[feature_columns].to_numpy(dtype=float)

    return reward_dict, context_dict, feature_columns
