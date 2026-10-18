"""
How much of the class confusion is irreducible.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import data


def _row_keys(X):
    """
    One 64-bit hash per row; equal rows hash equal.
    """

    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def _majority_label(frame):
    """
    The most common label of every distinct feature vector.
    """

    counts = frame.groupby(["key", "y"]).size().reset_index(name="n")
    counts = counts.sort_values(["key", "n", "y"], ascending=[True, False, True])
    return counts.drop_duplicates("key").set_index("key")["y"]


def label_conflicts(X, y):
    """
    Per-class duplication and conflict rates, an idealised ceiling, and the
    class-to-class overlap matrix.
    """

    codes, names = data.class_labels(y)
    frame = pd.DataFrame({"key": _row_keys(X), "y": y.to_numpy()})

    labels_per_key = frame.groupby("key")["y"].transform("nunique")
    duplicated = frame.duplicated("key", keep=False)
    conflicting = labels_per_key > 1

    ceiling_prediction = frame["key"].map(_majority_label(frame)).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["y"], ceiling_prediction, labels=codes, zero_division=0
    )

    per_class = pd.DataFrame({
        "class": names,
        "support": support,
        "share_duplicated": [duplicated[frame["y"] == c].mean() for c in codes],
        "share_conflicting": [conflicting[frame["y"] == c].mean() for c in codes],
        "ceiling_recall": recall,
        "ceiling_precision": precision,
        "ceiling_f1": f1,
    })

    overall = pd.DataFrame([{
        "instances": len(frame),
        "distinct_feature_vectors": frame["key"].nunique(),
        "share_duplicated": duplicated.mean(),
        "share_conflicting": conflicting.mean(),
        "ceiling_accuracy": accuracy_score(frame["y"], ceiling_prediction),
        "ceiling_f1_macro": f1_score(frame["y"], ceiling_prediction, average="macro"),
    }])

    return per_class, overall, _overlap(frame, codes, names)


def _overlap(frame, codes, names):
    """
    overlap[a, b]: share of class a rows whose exact feature vector also
    occurs labelled as class b.
    """

    rows_per_label = frame.groupby(["key", "y"]).size().reset_index(name="rows")
    pairs = rows_per_label.merge(rows_per_label[["key", "y"]], on="key", suffixes=("", "_other"))
    pairs = pairs[pairs["y"] != pairs["y_other"]]

    counts = pairs.groupby(["y", "y_other"])["rows"].sum().unstack(fill_value=0)
    counts = counts.reindex(index=codes, columns=codes, fill_value=0)
    support = frame["y"].value_counts().reindex(codes).to_numpy()

    shares = counts.to_numpy() / support[:, None]
    np.fill_diagonal(shares, np.nan)
    return pd.DataFrame(shares, index=names, columns=names)


def conflict_votes(X, y, predictions, class_weight):
    """
    Which label wins among rows that cannot be told apart, with and without
    class weights, and what each model predicted for those rows.
    """

    codes, names = data.class_labels(y)
    frame = pd.DataFrame({"key": _row_keys(X), "y": y.to_numpy()})
    conflicting = (frame.groupby("key")["y"].transform("nunique") > 1).to_numpy()

    groups = frame[conflicting].groupby(["key", "y"]).size().reset_index(name="n")
    groups["weighted"] = groups["n"] * groups["y"].map(class_weight)
    rows = frame[conflicting]

    table = pd.DataFrame(index=pd.Index(names, name="class"))
    table["conflicting_rows"] = rows["y"].value_counts().reindex(codes, fill_value=0).to_numpy()
    for column, score in [("won_unweighted", "n"), ("won_balanced", "weighted")]:
        winner = groups.sort_values([score, "y"]).drop_duplicates("key", keep="last")
        winner = rows["key"].map(winner.set_index("key")["y"])
        table[column] = winner.value_counts().reindex(codes, fill_value=0).to_numpy()
    for key, prediction in predictions.items():
        predicted = pd.Series(prediction[conflicting])
        table[f"predicted_{key}"] = predicted.value_counts().reindex(codes, fill_value=0).to_numpy()
    return table.reset_index()

