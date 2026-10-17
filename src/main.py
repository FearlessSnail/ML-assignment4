"""
Entry point: tunes a random forest on the UNSW-NB15 network traffic dataset
and compares it with the k-nearest neighbours and classification tree models of
assignment 2, on identical cross-validation folds.

    python src/main.py
    python src/main.py --replot # redraw the figures from a previous run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import data, diagnostics, experiment, forest_model, knn_model, plots, tree_model
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
    parser.add_argument("--output-dir", default=ROOT / "out", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--tune-sample", default=40000, type=int)
    parser.add_argument("--sample", default=0, type=int,
                        help="subsample the whole experiment (for a quick smoke test)")
    parser.add_argument("--refit-knn", action="store_true",
                        help="refit kNN instead of re-using the assignment 2 predictions")
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from a previous run without refitting")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--jobs", default=-1, type=int)
    return parser.parse_args()


def saved_knn_predictions(path, y, log):
    """
    The assignment 2 kNN predictions,
    """

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
    out = args.output_dir
    log = Progress()

    if args.replot:
        plots.plot_all(out)
        log(f"figures redrawn in {out}/")
        return

    out.mkdir(parents=True, exist_ok=True)

    # --- data --------- #

    log.rule("DATA")
    with Timer(log, f"loading {args.data}"):
        X, y = data.load_dataset(args.data, sample_size=args.sample or None,
                                 random_state=args.seed)
        
    codes, names = data.class_labels(y)
    (out / "class_names.json").write_text(json.dumps(dict(zip(map(str, codes), names))))
    log(f"{len(y)} instances, {X.shape[1]} features, {len(codes)} classes")

    with Timer(log, "measuring duplicated and conflicting feature vectors"):
        per_class, overall, overlap = diagnostics.label_conflicts(X, y)

    per_class.to_csv(out / "label_conflicts.csv", index=False)
    overall.to_csv(out / "ceiling.csv", index=False)
    overlap.to_csv(out / "label_overlap.csv")
    row = overall.iloc[0]
    log(f"{row['share_duplicated']:.1%} of rows repeat another row's features; "
        f"{row['share_conflicting']:.1%} share them with a row of another class")
    log(f"idealised ceiling: accuracy {row['ceiling_accuracy']:.4f}, "
        f"macro F1 {row['ceiling_f1_macro']:.4f}")

    # --- tuning -------- #

    log.rule("TUNING " + forest_model.NAME)    
    X_tune, y_tune = X, y
    if args.tune_sample and args.tune_sample < len(y):
        X_tune, y_tune = data.stratified_sample(X, y, args.tune_sample, args.seed)
    log(f"tuning on {len(y_tune)} instances (the assignment 2 tuning sample), "
        f"{args.folds}-fold inner CV", 1)

    params, stages = experiment.tune(forest_model, X_tune, y_tune, args.folds, args.seed, log,
                                     n_jobs=args.jobs)
    stages.to_csv(out / "tuning_stages.csv", index=False)

    log(f"stage {len(forest_model.SEARCH_STAGES) + 1} (Q1: number of trees): "
        f"growing {max(forest_model.TREE_COUNTS)} trees per fold", 1)
    inner_curve = experiment.tree_count_curve(
        forest_model, X_tune, y_tune, params, forest_model.TREE_COUNTS,
        experiment.make_cv(args.folds, args.seed), log)
    inner_curve.to_csv(out / "tuning_tree_count.csv", index=False)

    params["classifier__n_estimators"] = experiment.choose_tree_count(
        inner_curve, forest_model.PLATEAU_TOLERANCE)
    log(f"chose n_estimators={params['classifier__n_estimators']}", 2)
    log(f"selected configuration: {experiment.format_params(params)}", 1)
    (out / "selected_configuration.json").write_text(json.dumps(params, indent=2))

    # --- evaluation -------- #
    log.rule("EVALUATION")
    cv = experiment.make_cv(args.folds, args.seed)
    saved = None if args.refit_knn else saved_knn_predictions(args.a2_predictions, y, log)

    curve_counts = [m for m in forest_model.TREE_COUNTS if m <= params["classifier__n_estimators"]]
    if params["classifier__n_estimators"] not in curve_counts:
        curve_counts.append(params["classifier__n_estimators"])

    candidates = [
        experiment.Candidate("kNN", knn_model.NAME + " (A2)", knn_model,
                             {**KNN_PARAMS, "classifier__n_jobs": args.jobs},
                             saved_predictions=None if saved is None else saved["kNN"].to_numpy()),
        experiment.Candidate("Tree", tree_model.NAME + " (A2)", tree_model, TREE_PARAMS),
        experiment.Candidate("RF-untuned", forest_model.NAME + " (untuned)", forest_model,
                             forest_model.UNTUNED_PARAMS, n_jobs=args.jobs),
        experiment.Candidate("RF", forest_model.NAME + " (tuned)", forest_model, params,
                             n_jobs=args.jobs, curve_counts=curve_counts),
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

    experiment.finish(results, y, codes, names)
    pairs = [("RF", "Tree"), ("RF", "kNN"), ("RF", "RF-untuned")]
    comparisons = [experiment.compare(results[a], results[b], a, b, len(y)) for a, b in pairs]

    log.rule("OUTPUT")
    with Timer(log, "writing tables and figures"):
        experiment.save(results, comparisons, out)
        plots.plot_all(out)

    print(experiment.report(results, comparisons, args.folds, len(y)))
    log(f"total runtime {log.elapsed() / 60:.1f} minutes; results in {out}/")


if __name__ == "__main__":
    main()
