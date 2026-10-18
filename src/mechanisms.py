"""
Analysis on why forests beat single treas.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import data



def decompose(predictions, y_true):
    """
    Bias and variance of 0-1 loss (Domingos, 2000), for every test row.
    """

    main = stats.mode(predictions, axis=0, keepdims=False).mode
    return pd.DataFrame({
        "y": y_true,
        "loss": (predictions != y_true).mean(axis=0),
        "bias": (main != y_true).astype(float),
        "variance": (predictions != main).mean(axis=0),
    })


def _summarise_rows(rows):
    biased = rows["bias"] == 1
    return {
        "loss": rows["loss"].mean(),
        "bias": rows["bias"].mean(),
        "variance": rows["variance"].mean(),
        "variance_where_unbiased": (rows["variance"] * ~biased).mean(),
        "variance_where_biased": (rows["variance"] * biased).mean(),
        "net_variance": rows["loss"].mean() - rows["bias"].mean(),
    }


def bias_variance(candidates, X, y, cv, repeats, fraction, seed, log):
    """
    Q5: train every candidate on `repeats` stratified subsamples of each outer
    training fold, and decompose its error on that fold's test rows.
    """

    codes, names = data.class_labels(y)
    summary, per_class = [], []

    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        y_test = y.iloc[test_index].to_numpy()
        subsets = [
            train_test_split(train_index, train_size=fraction, stratify=y.iloc[train_index],
                             random_state=seed + repeat)[0]
            for repeat in range(repeats)
        ]

        for candidate in candidates:
            started = time.time()
            predictions = np.empty((repeats, len(test_index)), dtype=int)
            for repeat, subset in enumerate(subsets):
                threads = {} if candidate.n_jobs is None else {"n_jobs": candidate.n_jobs}
                model = candidate.module.build_model(X, **threads, **candidate.params)
                model.fit(X.iloc[subset], y.iloc[subset])
                predictions[repeat] = model.predict(X.iloc[test_index])
                del model

            rows = decompose(predictions, y_test)
            by_class = {name: _summarise_rows(rows[rows["y"] == code])
                        for code, name in zip(codes, names)}
            class_mean = pd.DataFrame(by_class).T.mean()

            summary.append({"model": candidate.key, "fold": fold, "scope": "all rows",
                            **_summarise_rows(rows)})
            summary.append({"model": candidate.key, "fold": fold, "scope": "class average",
                            **class_mean.to_dict()})
            for name, values in by_class.items():
                per_class.append({"model": candidate.key, "fold": fold, "class": name, **values})

            log(f"fold {fold + 1}, {candidate.key}: loss {summary[-2]['loss']:.4f}, "
                f"bias {summary[-2]['bias']:.4f}, variance {summary[-2]['variance']:.4f} "
                f"({time.time() - started:.0f}s)", 2)

    return pd.DataFrame(summary), pd.DataFrame(per_class)


def strength_and_correlation(tree_votes, y_index):
    """
    Breiman's (2001, section 2.2) strength and correlation, from the hard votes
    of every tree on held-out rows.
    """

    trees, rows = tree_votes.shape
    classes = int(max(tree_votes.max(), y_index.max())) + 1
    share = np.zeros((rows, classes))
    for votes in tree_votes:
        share[np.arange(rows), votes] += 1
    share /= trees

    rival_share = share.copy()
    rival_share[np.arange(rows), y_index] = -np.inf
    rival = rival_share.argmax(axis=1)

    margin = share[np.arange(rows), y_index] - share[np.arange(rows), rival]
    strength = margin.mean()

    p_true = (tree_votes == y_index).mean(axis=1)
    p_rival = (tree_votes == rival).mean(axis=1)
    spread = np.sqrt(p_true + p_rival - (p_true - p_rival) ** 2).mean()
    correlation = margin.var() / spread ** 2

    bound = correlation * (1 - strength ** 2) / strength ** 2 if strength > 0 else np.nan
    return strength, correlation, bound


def feature_subset_mechanism(module, X, y, params, feature_values, trees, cv, log, n_jobs=-1):
    """
    Q6: for every feature-subset size, fit a forest under cross-validation and
    record its strength, correlation, Breiman's bound, the macro F1 of its
    individual trees and the macro F1 of the forest.
    """

    rows = []
    for value in feature_values:
        started = time.time()
        setting = {**params, "classifier__max_features": value,
                   "classifier__n_estimators": trees}

        for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
            model = module.build_model(X, n_jobs=n_jobs, **setting)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            forest = model.named_steps["classifier"]
            features = np.asarray(model.named_steps["preprocess"].transform(X.iloc[test_index]),
                                  dtype=np.float32)
            y_test = y.iloc[test_index].to_numpy()
            y_index = np.searchsorted(forest.classes_, y_test)

            tree_votes = np.vstack([tree.predict_proba(features).argmax(axis=1)
                                    for tree in forest.estimators_])
            strength, correlation, bound = strength_and_correlation(tree_votes, y_index)
            tree_f1 = np.mean([f1_score(y_index, votes, average="macro", zero_division=0)
                               for votes in tree_votes])

            rows.append({
                "max_features": "none" if value is None else str(value),
                "fold": fold,
                "strength": strength,
                "correlation": correlation,
                "breiman_bound": bound,
                "tree_accuracy": (tree_votes == y_index).mean(),
                "tree_f1_macro": tree_f1,
                "forest_f1_macro": f1_score(y_test, model.predict(X.iloc[test_index]),
                                            average="macro", zero_division=0),
            })

        mean = pd.DataFrame(rows[-cv.get_n_splits():]).mean(numeric_only=True)
        log(f"max_features={value}: strength {mean['strength']:.3f}, correlation "
            f"{mean['correlation']:.3f}, forest macro F1 {mean['forest_f1_macro']:.4f} "
            f"({time.time() - started:.0f}s)", 2)

    return pd.DataFrame(rows)

