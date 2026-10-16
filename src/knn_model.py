"""k-nearest neighbours model, with the pre-processing that this algorithm needs.

kNN classifies by Euclidean/Manhattan distance in the feature space, so it is
sensitive to feature scale, to skewed heavy-tailed features, and to redundant
features (which are effectively counted twice in the distance). The pipeline
below addresses exactly those three issues and nothing else.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

import data

NAME = "k-Nearest Neighbours"
SHORT_NAME = "kNN"

REDUNDANT_FEATURES = [
    "spkts",
    "sloss",
    "dpkts",
    "dloss",
    "dwin",
    "ct_dst_src_ltm",
    "ct_srv_dst",
    "ct_src_dport_ltm",
    "ct_ftp_cmd",
]


MIN_LEVEL_FREQUENCY = 0.01


def log_transform(X):
    return np.log1p(np.clip(X, 0, None))


def to_float32(X):
    return np.asarray(X, dtype=np.float32)


def build_preprocessor(X):
    numeric = [c for c in data.numeric_features(X) if c not in REDUNDANT_FEATURES]
    nominal = data.nominal_features(X)

    numeric_pipeline = Pipeline(
        [
            ("log", FunctionTransformer(log_transform, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )

    nominal_encoder = OneHotEncoder(
        min_frequency=MIN_LEVEL_FREQUENCY,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )

    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("nominal", nominal_encoder, nominal)],
        remainder="drop",
    )


def build_model(X, **params):
    """
    Full kNN pipeline. Pre-processing is fitted inside cross-validation folds.
    """

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("cast", FunctionTransformer(to_float32)),
            ("classifier", KNeighborsClassifier(algorithm="brute", n_jobs=1)),
        ]
    )
    
    if params:
        pipeline.set_params(**params)
    return pipeline


SEARCH_STAGES = [
    {
        "preprocess__numeric__scale": [StandardScaler(), RobustScaler()],
        "classifier__p": [1, 2],
    },
    {
        "classifier__n_neighbors": [1, 3, 5, 9, 15, 25],
        "classifier__weights": ["uniform", "distance"],
    },
]

DEFAULT_PARAMS = {
    "classifier__n_neighbors": 5,
    "classifier__weights": "uniform",
    "classifier__p": 2,
}
