"""
Analysis on why forests beat single treas.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats
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

