"""
Tuning of the random forest and its comparison with the assignment 2 models.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import data

PRIMARY_METRIC = "f1_macro"


def scorers(codes):
    """
    The assignment 2 measures, plus the F1 of every class on its own so that the
    tuning can show which categories a parameter helps or hurts.
    """

    table = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "mcc": make_scorer(matthews_corrcoef),
    }
    for code in codes:
        table[f"f1_class_{code}"] = make_scorer(
            f1_score, labels=[code], average="macro", zero_division=0
        )
    return table


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


def format_params(params):
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in sorted(params.items()))


# ------ tuning ---------------- #


def tune(module, X, y, folds, seed, log, n_jobs=-1):
    """
    Staged grid search over module.SEARCH_STAGES.

    Each stage fixes the winners of the stages before it. The forests inside the
    search run single-threaded so that the candidates run in parallel instead.
    """

    codes, _ = data.class_labels(y)
    params = dict(module.DEFAULT_PARAMS)
    tables = []

    for number, stage in enumerate(module.SEARCH_STAGES, start=1):
        grid = stage["grid"]
        candidates = math.prod(len(values) for values in grid.values())
        log(f"stage {number}/{len(module.SEARCH_STAGES)} ({stage['question']}): "
            f"{candidates} candidates, {candidates * folds} fits", 1)

        search = GridSearchCV(
            estimator=module.build_model(X, n_jobs=1, **params),
            param_grid=grid,
            scoring=scorers(codes),
            refit=False,
            cv=make_cv(folds, seed),
            n_jobs=n_jobs,
        )

        started = time.time()
        search.fit(X, y)
        results = pd.DataFrame(search.cv_results_)
        best = results[f"mean_test_{PRIMARY_METRIC}"].idxmax()
        chosen = results.loc[best, "params"]
        params.update(chosen)

        log(f"chose {format_params(chosen)} (macro F1 "
            f"{results.loc[best, f'mean_test_{PRIMARY_METRIC}']:.4f}, "
            f"took {time.time() - started:.0f}s)", 2)

        results["stage"] = number
        results["question"] = stage["question"]
        keep = ["stage", "question"] + [
            c for c in results.columns if c.startswith(("param_", "mean_", "std_"))
        ]
        tables.append(results[keep])

    return params, pd.concat(tables, ignore_index=True)


def staged_predictions(model, X, counts):
    """
    Predictions of the forests made of the first m trees, for every m in counts.
    """

    forest = model.named_steps["classifier"]
    features = np.asarray(model.named_steps["preprocess"].transform(X), dtype=np.float32)
    total = np.zeros((len(features), forest.n_classes_))
    wanted = set(counts)

    for m, tree in enumerate(forest.estimators_, start=1):
        total += tree.predict_proba(features)
        if m in wanted:
            yield m, forest.classes_[total.argmax(axis=1)]


def tree_count_curve(module, X, y, params, counts, cv, log):
    """
    Q1: macro F1 against the number of trees, under cross-validation.
    """

    params = {**params, "classifier__n_estimators": max(counts)}
    rows = []

    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        started = time.time()
        model = module.build_model(X, n_jobs=-1, **params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        y_test = y.iloc[test_index]

        for m, prediction in staged_predictions(model, X.iloc[test_index], counts):
            rows.append({"fold": fold, "trees": m, **fold_metrics(y_test, prediction)})
        log(f"fold {fold + 1}: {max(counts)} trees grown and scored in "
            f"{time.time() - started:.0f}s", 2)

    return pd.DataFrame(rows)


def choose_tree_count(curve, tolerance):
    """
    The smallest forest whose mean macro F1 is within tolerance of the largest.
    """

    mean = curve.groupby("trees")[PRIMARY_METRIC].mean()
    target = mean.loc[mean.index.max()] - tolerance
    return int(mean[mean >= target].index.min())


# --- evaluation ------- #


@dataclass
class Candidate:
    """
    One model in the final comparison.
    """

    key: str
    name: str
    module: object = None
    params: dict = field(default_factory=dict)
    n_jobs: int = None
    curve_counts: list = None
    saved_predictions: np.ndarray = None


def evaluate(candidate, X, y, cv, log):
    """
    Outer cross-validation, folds run one after another.
    """

    predictions = np.empty(len(y), dtype=int)
    fold_index = np.empty(len(y), dtype=int)
    rows, timing, curve = [], [], []

    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        y_test = y.iloc[test_index]
        fold_index[test_index] = fold

        if candidate.saved_predictions is not None:
            fold_predictions = candidate.saved_predictions[test_index]
            fit_time = predict_time = np.nan
        else:
            threads = {} if candidate.n_jobs is None else {"n_jobs": candidate.n_jobs}
            model = candidate.module.build_model(X, **threads, **candidate.params)
            started = time.time()
            model.fit(X.iloc[train_index], y.iloc[train_index])
            fit_time = time.time() - started

            started = time.time()
            fold_predictions = model.predict(X.iloc[test_index])
            predict_time = time.time() - started

            if candidate.curve_counts:
                for m, staged in staged_predictions(model, X.iloc[test_index],
                                                    candidate.curve_counts):
                    curve.append({"fold": fold, "trees": m, **fold_metrics(y_test, staged)})
            del model

        predictions[test_index] = fold_predictions
        rows.append(fold_metrics(y_test, fold_predictions))
        timing.append({"fold": fold, "fit_seconds": fit_time, "predict_seconds": predict_time})
        log(f"fold {fold + 1}: macro F1 {rows[-1][PRIMARY_METRIC]:.4f}"
            + ("" if np.isnan(fit_time) else f"  (fit {fit_time:.0f}s, predict {predict_time:.1f}s)"),
            2)

    scores = pd.DataFrame(rows)
    scores.index.name = "fold"
    return {
        "name": candidate.name,
        "scores": scores,
        "summary": pd.DataFrame({"mean": scores.mean(), "std": scores.std(ddof=1)}),
        "predictions": predictions,
        "fold_index": fold_index,
        "timing": pd.DataFrame(timing),
        "curve": pd.DataFrame(curve),
    }


def compare(result_a, result_b, key_a, key_b, metric=PRIMARY_METRIC):
    """
    Paired comparison of two models on the same outer folds, as in assignment 2.
    """

    a = result_a["scores"][metric].to_numpy()
    b = result_b["scores"][metric].to_numpy()
    difference = a - b

    t_stat, t_p = stats.ttest_rel(a, b)
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

    return {
        "model_a": key_a,
        "model_b": key_b,
        "metric": metric,
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mean_difference": difference.mean(),
        "cohens_d": difference.mean() / pooled if pooled > 0 else np.nan,
        "folds_favouring_a": int((difference > 0).sum()),
        "paired_t": t_stat,
        "paired_t_p": t_p,
    }


def report(results, comparisons, folds, n_instances):
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

    lines.append("\nPairwise comparisons on macro F1 (positive difference favours the first model)")
    for row in comparisons:
        lines.append(
            f"  {row['model_a'] + ' vs ' + row['model_b']:<22} diff {row['mean_difference']:+.4f}  "
            f"d = {row['cohens_d']:+.2f}  {row['folds_favouring_a']}/{folds} folds  "
            f"paired p = {row['paired_t_p']:.1e}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)
