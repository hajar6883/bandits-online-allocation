from __future__ import annotations

from pathlib import Path

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


def load_raw(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    tables = {
        "small_matrix": pd.read_csv(path / "small_matrix.csv"),
        "item_categories": pd.read_csv(path / "item_categories.csv"),
        "user_features": pd.read_csv(path / "user_features.csv"),
        "item_daily_feat": pd.read_csv(path / "item_daily_features.csv"),
    }
    tables["item_categories"]["feat"] = tables["item_categories"]["feat"].map(eval)
    return tables


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


def watch_ratio_to_reward(watch_ratios: list[float]) -> list[int]:
    return [1 if w > WATCH_THRESHOLD else 0 for w in watch_ratios]


def build_reward_dict(df: pd.DataFrame) -> dict[int, list[int]]:
    raw = (
        df.groupby("video_id")["watch_ratio"]
        .apply(list)
        .apply(watch_ratio_to_reward)
        .to_dict()
    )
    return {i: rewards for i, (_, rewards) in enumerate(raw.items())}


def build_context_dict(df: pd.DataFrame) -> dict[int, np.ndarray]:
    df1 = df.drop(["user_id", "watch_ratio"], axis=1)
    raw = df1.groupby("video_id").apply(
        lambda x: x.iloc[:, 1:].values.tolist()
    ).to_dict()
    return {i: np.array(ctx) for i, (_, ctx) in enumerate(raw.items())}


def prepare(path: str | Path) -> tuple[pd.DataFrame, dict[int, list[int]], dict[int, np.ndarray]]:
    tables = load_raw(path)
    df = build_subsample(tables["small_matrix"], tables["user_features"])
    reward_dict = build_reward_dict(df)
    context_dict = build_context_dict(df)
    return df, reward_dict, context_dict


def build_one_hot_items(item_categories: pd.DataFrame, n_tags: int = 32) -> pd.DataFrame:
    rows = np.zeros((len(item_categories), n_tags), dtype=int)
    for i, feats in enumerate(item_categories["feat"].values):
        for f in feats:
            rows[i, f] = 1
    cols = [f"feat_video{i}" for i in range(n_tags)]
    one_hot = pd.DataFrame(rows, columns=cols)
    return pd.concat([item_categories[["video_id"]].reset_index(drop=True), one_hot], axis=1)


def exp_score_generator(df: pd.DataFrame, n_tags: int = 32) -> dict[int, float]:
    liked = df[df["watch_ratio"] >= 1]
    tag_columns = [f"feat_video{i}" for i in range(n_tags)]
    user_score = liked.groupby("user_id")[tag_columns].sum()
    user_score["exp_score"] = user_score.astype(bool).sum(axis=1) / n_tags

    scaler = MinMaxScaler()
    user_score["exp_score"] = scaler.fit_transform(user_score[["exp_score"]].values)

    user_score = user_score[["exp_score"]].reset_index()
    return dict(zip(user_score["user_id"], user_score["exp_score"]))
