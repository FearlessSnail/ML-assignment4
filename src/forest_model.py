"""
Random forest model, built on the classification tree of assignment 2.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import tree_model

NAME = "Random Forest"
SHORT_NAME = "RF"


def build_preprocessor(X):
    return tree_model.build_preprocessor(X)


def build_model(X, random_state=42, n_jobs=1, **params):
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)),
        ]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline


# number of trees used while the other parameters are searched.
SEARCH_TREES = 200


SEARCH_STAGES = [
    {
        "question": "Q2+Q4: feature subset size and class weighting",

        "grid": {
            "classifier__max_features": [1, "sqrt", 0.25, 0.5, None],
            "classifier__class_weight": [None, "balanced", "balanced_subsample"],
        },
    },
    {
        "question": "Q3: size of the individual trees",
        "grid": {
            "classifier__min_samples_leaf": [1, 2, 5, 10, 25],
            "classifier__criterion": ["gini", "entropy"],
        },
    },
]


TREE_COUNTS = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500]

PLATEAU_TOLERANCE = 0.002

DEFAULT_PARAMS = {
    "classifier__n_estimators": SEARCH_TREES,
    "classifier__max_features": "sqrt",
    "classifier__min_samples_leaf": 1,
    "classifier__class_weight": None,
    "classifier__criterion": "gini",
}

# scikit-learn's out-of-the-box forest, evaluated alongside the tuned one so
# that the value added by the tuning itself can be measured.
UNTUNED_PARAMS = {
    "classifier__n_estimators": 100,
    "classifier__max_features": "sqrt",
    "classifier__min_samples_leaf": 1,
    "classifier__class_weight": None,
    "classifier__criterion": "gini",
}


VARIANCE_REPEATS = 10
VARIANCE_FRACTION = 0.5

STRENGTH_FEATURES = [1, "sqrt", 0.25, 0.5, None]
STRENGTH_TREES = 100
