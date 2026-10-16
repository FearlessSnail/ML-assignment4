"""Entry point: re-scores the k-nearest neighbours and classification tree
models of assignment 2 on the UNSW-NB15 dataset, on the same five folds, as the
baseline for the ensemble study.

    python src/main.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import data, experiment, knn_model, tree_model
from progress import Progress, Timer

ROOT = Path(__file__).resolve().parent.parent

TREE_PARAMS = {
    "classifier__criterion": "gini",
    "classifier__class_weight": None,
    "classifier__max_depth": None,
    "classifier__min_samples_leaf": 1,
    "classifier__ccp_alpha": 1e-4,
}

KNN_PARAMS = {
    "preprocess__numeric__scale": RobustScaler(),
    "classifier__p": 1,
    "classifier__n_neighbors": 3,
    "classifier__weights": "distance",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ROOT / "data" / "networkTraffic.csv", type=Path)
    parser.add_argument("--a2-predictions", type=Path,
                        default=ROOT / "data" / "assignment2_predictions.csv",
                        help="out-of-fold predictions saved by the assignment 2 run")
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--sample", default=0, type=int,
                        help="subsample the whole experiment (for a quick smoke test)")
    parser.add_argument("--refit-knn", action="store_true",
                        help="refit kNN instead of re-using the assignment 2 predictions")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--jobs", default=-1, type=int)
    return parser.parse_args()


def saved_knn_predictions(path, y, log):

    if not path.exists():
        log(f"{path} not found; kNN will be refitted", 1)
        return None
    saved = pd.read_csv(path)
    if len(saved) != len(y) or not (saved["actual"].to_numpy() == y.to_numpy()).all():
        log("saved predictions do not match this dataset; kNN will be refitted", 1)
        return None
    return saved


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    log = Progress()

    # --- data --------- #
    log.rule("DATA")
    with Timer(log, f"loading {args.data}"):
        X, y = data.load_dataset(args.data, sample_size=args.sample or None,
                                 random_state=args.seed)
    codes, _ = data.class_labels(y)
    log(f"{len(y)} instances, {X.shape[1]} features, {len(codes)} classes")

    # --- evaluation -------- #
    log.rule("EVALUATION")
    cv = experiment.make_cv(args.folds, args.seed)
    saved = None if args.refit_knn else saved_knn_predictions(args.a2_predictions, y, log)

    candidates = [
        experiment.Candidate("kNN", knn_model.NAME + " (A2)", knn_model,
                             {**KNN_PARAMS, "classifier__n_jobs": args.jobs},
                             saved_predictions=None if saved is None else saved["kNN"].to_numpy()),
        experiment.Candidate("Tree", tree_model.NAME + " (A2)", tree_model, TREE_PARAMS),
    ]

    results = {}
    for candidate in candidates:
        source = "assignment 2 predictions" if candidate.saved_predictions is not None else "fit"
        log(f"{candidate.name}: outer {args.folds}-fold CV ({source})", 1)
        results[candidate.key] = experiment.evaluate(candidate, X, y, cv, log)

    if saved is not None:
        agreement = np.mean(results["Tree"]["predictions"] == saved["Tree"].to_numpy())
        log(f"reproduction check: refitted tree agrees with assignment 2 on {agreement:.2%} "
            f"of predictions", 1)

    print(experiment.report(results, args.folds, len(y)))
    log(f"total runtime {log.elapsed() / 60:.1f} minutes")


if __name__ == "__main__":
    main()
