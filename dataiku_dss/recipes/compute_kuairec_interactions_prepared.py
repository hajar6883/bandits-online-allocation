"""Step 1: clean and join the KuaiRec interaction, user and item tables."""

import dataiku

from kuairec_prep import build_subsample, join_item_features

variables = dataiku.get_custom_variables()
keep_video_frac = float(variables.get("keep_video_frac", 0.025))
keep_user_frac = float(variables.get("keep_user_frac", 0.1))

small_matrix = dataiku.Dataset("kuairec_small_matrix").get_dataframe()
user_features = dataiku.Dataset("kuairec_user_features").get_dataframe()
item_categories = dataiku.Dataset("kuairec_item_categories").get_dataframe()

print("small_matrix   : %d rows" % len(small_matrix))
print("user_features  : %d rows" % len(user_features))
print("item_categories: %d rows" % len(item_categories))

interactions = build_subsample(
    small_matrix,
    user_features,
    keep_video_frac=keep_video_frac,
    keep_user_frac=keep_user_frac,
)
print("after sub-sampling: %d rows, %d videos, %d users" % (
    len(interactions),
    interactions["video_id"].nunique(),
    interactions["user_id"].nunique(),
))

prepared = join_item_features(interactions, item_categories)
print("after item join   : %d rows, %d columns" % prepared.shape)

dataiku.Dataset("kuairec_interactions_prepared").write_with_schema(prepared)
