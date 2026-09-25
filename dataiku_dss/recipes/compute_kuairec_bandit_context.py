"""Step 2: build the Bernoulli reward and the per-arm context matrix."""

import dataiku

from kuairec_prep import (
    build_context_frame,
    context_schema,
    exp_score_generator,
)

variables = dataiku.get_custom_variables()
include_item_features = str(
    variables.get("include_item_features", "false")
).strip().lower() in ("1", "true", "yes")

prepared = dataiku.Dataset("kuairec_interactions_prepared").get_dataframe()
print("prepared: %d rows, %d columns" % prepared.shape)

context = build_context_frame(prepared, include_item_features=include_item_features)
schema = context_schema(prepared, include_item_features=include_item_features)

n_arms = int(context["arm_id"].nunique())
n_features = len(schema)
print("n_arms = %d, context dim = %d" % (n_arms, n_features))
print("mean logged reward = %.4f" % context["reward"].mean())

user_profile = exp_score_generator(prepared)
print("user_profile: %d users" % len(user_profile))

dataiku.Dataset("kuairec_bandit_context").write_with_schema(context)
dataiku.Dataset("kuairec_context_schema").write_with_schema(schema)
dataiku.Dataset("kuairec_user_profile").write_with_schema(user_profile)
