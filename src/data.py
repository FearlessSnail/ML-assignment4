"""
Reading and cleaning of the UNSW-NB15 network traffic dataset.
"""

import pandas as pd

TARGET = "attack_cat"
ID_COLUMN = "id"
NOMINAL_FEATURES = ["proto", "service", "state"]

BINARY_FEATURES = ["is_ftp_login", "is_sm_ips_ports"]

MISSING_TOKEN = "?"
MISSING_CATEGORY = "none"

CLASS_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Analysis",
    6: "Fuzzers",
    7: "Worms",
    8: "Shellcode",
    9: "Generic",
}


def load_raw(path):
    """
    Read the CSV, treating '?' as missing.
    """
    return pd.read_csv(path, na_values=[MISSING_TOKEN], low_memory=False)


def clean(df, drop_duplicates=False):
    """
    Apply the algorithm-independent cleaning steps and split off the target
    """
    df = df.copy()

    # 'id' is a unique row identifier, so it carries no generalisable signal.
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    df = df.dropna(subset=[TARGET])

    if drop_duplicates:
        df = df.drop_duplicates()

    for column in NOMINAL_FEATURES:
        if column in df.columns:

            df[column] = df[column].fillna(MISSING_CATEGORY).astype(str).str.strip().str.lower()

    for column in BINARY_FEATURES:
        if column in df.columns:
            df[column] = df[column].clip(0, 1)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])


    numeric_columns = [c for c in X.columns if c not in NOMINAL_FEATURES]
    X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors="coerce")
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())

    # constant features cannot separate any pair of classes.
    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)

    return X, y


def load_dataset(path, drop_duplicates=False, sample_size=None, random_state=42):
    """
    Read, clean and optionally take a class-stratified subsample.
    """

    X, y = clean(load_raw(path), drop_duplicates=drop_duplicates)

    if sample_size is not None and sample_size < len(y):
        X, y = stratified_sample(X, y, sample_size, random_state)
    return X, y


def stratified_sample(X, y, size, random_state=42):
    """
    Subsample while preserving the class distribution
    """

    from sklearn.model_selection import train_test_split

    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=size, stratify=y, random_state=random_state
    )
    return X_sample, y_sample


def numeric_features(X):
    return [c for c in X.columns if c not in NOMINAL_FEATURES]


def nominal_features(X):
    return [c for c in X.columns if c in NOMINAL_FEATURES]


def class_labels(y):
    """
    Class codes present in y, with their names.
    """

    codes = sorted(pd.unique(y))
    return codes, [CLASS_NAMES.get(c, str(c)) for c in codes]


def correlated_pairs(X, threshold=0.95):
    """
    Pairs of numeric features whose absolute correlation reaches threshold
    """

    numeric = X[numeric_features(X)]
    correlation = numeric.corr().abs().to_numpy()
    columns = list(numeric.columns)
    pairs = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if correlation[i, j] >= threshold:
                pairs.append((columns[i], columns[j], correlation[i, j]))
                
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"])


def summarise(X, y):


    codes, names = class_labels(y)
    counts = y.value_counts().reindex(codes)

    distribution = pd.DataFrame(
        {"class": names, "count": counts.to_numpy(), "proportion": (counts / len(y)).to_numpy()}
    )

    skewness = X[numeric_features(X)].skew().sort_values(ascending=False)
    ranges = X[numeric_features(X)].agg(["min", "max", "mean", "std"]).T
    
    return {
        "n_instances": len(y),
        "n_features": X.shape[1],
        "class_distribution": distribution,
        "imbalance_ratio": counts.max() / counts.min(),
        "skewness": skewness,
        "ranges": ranges,
        "correlated_pairs": correlated_pairs(X),
        "cardinality": X[nominal_features(X)].nunique(),
    }
