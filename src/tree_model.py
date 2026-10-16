"""
Classification tree model, with the pre-processing that this algorithm needs.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

import data

NAME = "Classification Tree"
SHORT_NAME = "Tree"


def build_preprocessor(X):
    numeric = data.numeric_features(X)
    nominal = data.nominal_features(X)

    nominal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("nominal", nominal_encoder, nominal), ("numeric", "passthrough", numeric)],
        remainder="drop",
    )


def build_model(X, random_state=42, **params):
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("classifier", DecisionTreeClassifier(random_state=random_state)),
        ]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline



SEARCH_STAGES = [
    {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier__max_depth": [None, 20],
        "classifier__min_samples_leaf": [1, 5, 20],
        "classifier__ccp_alpha": [0.0, 1e-5, 1e-4],
    },
]

DEFAULT_PARAMS = {
    "classifier__criterion": "gini",
    "classifier__max_depth": None,
    "classifier__min_samples_leaf": 1,
    "classifier__ccp_alpha": 0.0,
    "classifier__class_weight": None,
}


def describe_fitted_tree(pipeline):
    tree = pipeline.named_steps["classifier"]
    return {"depth": tree.get_depth(), "leaves": tree.get_n_leaves()}
