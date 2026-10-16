"""
Cross-validated evaluation of the assignment 2 models.

Every model is scored on the same five stratified folds as assignment 2, so
that any model added later can be compared with them fold by fold.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

PRIMARY_METRIC = "f1_macro"


def fold_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def make_cv(folds, seed):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


@dataclass
class Candidate:
    """
    One model in the comparison.
    """

    key: str
    name: str
    module: object = None
    params: dict = field(default_factory=dict)
    saved_predictions: np.ndarray = None


def evaluate(candidate, X, y, cv, log):
    """
    Outer cross-validation, folds run one after another.
    """

    predictions = np.empty(len(y), dtype=int)
    fold_index = np.empty(len(y), dtype=int)
    rows = []

    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        y_test = y.iloc[test_index]
        fold_index[test_index] = fold

        if candidate.saved_predictions is not None:
            fold_predictions = candidate.saved_predictions[test_index]
        else:
            model = candidate.module.build_model(X, **candidate.params)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            fold_predictions = model.predict(X.iloc[test_index])

        predictions[test_index] = fold_predictions
        rows.append(fold_metrics(y_test, fold_predictions))
        log(f"fold {fold + 1}: macro F1 {rows[-1][PRIMARY_METRIC]:.4f}", 2)

    scores = pd.DataFrame(rows)
    scores.index.name = "fold"
    return {
        "name": candidate.name,
        "scores": scores,
        "summary": pd.DataFrame({"mean": scores.mean(), "std": scores.std(ddof=1)}),
        "predictions": predictions,
        "fold_index": fold_index,
    }


def report(results, folds, n_instances):
    """
    Human-readable summary printed at the end of a run.
    """

    lines = ["", "=" * 78,
             f"RESULTS  ({folds}-fold stratified cross-validation, n = {n_instances})", "=" * 78,
             f"\n{'metric':<18}" + "".join(f"{key:>16}" for key in results)]

    first = next(iter(results.values()))
    for metric in first["summary"].index:
        cells = "".join(
            f"{r['summary'].loc[metric, 'mean']:>9.4f}±{r['summary'].loc[metric, 'std']:.4f}"
            for r in results.values()
        )
        lines.append(f"{metric:<18}{cells}")

    lines.append("=" * 78)
    return "\n".join(lines)
