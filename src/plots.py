"""
Figures, drawn from the CSV tables written by a run
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

DECOMPOSITION_COLOURS = {"loss": "#a3a29c", "bias": "#104281", "variance": "#86b6ef"}

MODEL_COLOURS = {"kNN": SERIES[0], "Tree": SERIES[1], "RF": SERIES[2]}
WEIGHT_COLOURS = {"none": SERIES[0], "balanced": SERIES[1], "balanced_subsample": SERIES[2]}
MODEL_NAMES = {
    "kNN": "k-nearest neighbours (A2)",
    "Tree": "Classification tree (A2)",
    "RF-untuned": "Random forest, untuned",
    "RF": "Random forest, tuned",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def _save(fig, output_dir, name):
    fig.savefig(output_dir / name)
    plt.close(fig)


def _class_order(per_class):
    return (per_class.drop_duplicates("class")
            .sort_values("support", ascending=False)["class"].tolist())


def _param_text(values):
    """
    Parameter values as text, with None (written to CSV as an empty cell) as "none".
    """

    values = pd.Series(values, dtype=object)
    return values.where(values.notna(), "none").map(lambda v: "none" if v is None else str(v))


def _feature_label(value):
    labels = {"1": "1 feature", "sqrt": "sqrt (6)", "0.25": "25% (10)",
              "0.5": "50% (21)", "none": "all 42\n(bagging)"}
    return labels.get(value, value)


def _weight_label(value):
    return value.replace("_", " ")


# --- data ------- #


def label_conflicts(output_dir):
    table = pd.read_csv(output_dir / "label_conflicts.csv").sort_values("support")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(table["class"], table["share_conflicting"], color=SERIES[0], height=0.65)
    for position, (share, support) in enumerate(zip(table["share_conflicting"], table["support"])):
        ax.annotate(f"{share:.0%}  (n={support:,})", (share, position), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)

    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("share of rows identical to a row with a different label")
    ax.grid(axis="y", visible=False)
    _save(fig, output_dir, "fig_label_conflicts.png")


# --- tuning -------- #


def tuning_features_and_weights(output_dir):
    """
    Q2 and Q4: macro F1 against feature subset size, one line per class weighting.
    """

    stage = pd.read_csv(output_dir / "tuning_stages.csv")
    stage = stage[stage["stage"] == 1].copy()
    stage["features"] = _param_text(stage["param_classifier__max_features"]).to_numpy()
    stage["weights"] = _param_text(stage["param_classifier__class_weight"]).to_numpy()
    order = list(dict.fromkeys(stage["features"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for weights, group in stage.groupby("weights", sort=False):
        group = group.set_index("features").loc[order]
        ax.errorbar(range(len(order)), group["mean_test_f1_macro"], yerr=group["std_test_f1_macro"],
                    color=WEIGHT_COLOURS[weights], marker="o", capsize=3,
                    label=_weight_label(weights))

    ax.set_xticks(range(len(order)), [_feature_label(v) for v in order])
    ax.set_xlabel("features considered at each split")
    ax.set_ylabel("macro F1 (inner CV)")
    ax.legend(title="class weighting", ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, output_dir, "fig_tuning_features_weights.png")


def tuning_weights_per_class(output_dir):
    """
    Q4: per-class F1 change from re-weighting, at the selected feature subset size.
    """

    stage = pd.read_csv(output_dir / "tuning_stages.csv")
    stage = stage[stage["stage"] == 1].copy()
    selected = json.loads((output_dir / "selected_configuration.json").read_text())
    chosen = _param_text([selected["classifier__max_features"]])[0]
    stage = stage[_param_text(stage["param_classifier__max_features"]).to_numpy() == chosen].copy()
    stage["weights"] = _param_text(stage["param_classifier__class_weight"]).to_numpy()
    names = json.loads((output_dir / "class_names.json").read_text())
    columns = [f"mean_test_f1_class_{code}" for code in names]

    baseline = stage[stage["weights"] == "none"][columns].iloc[0]
    weighted = stage[stage["weights"] != "none"]
    support = pd.read_csv(output_dir / "label_conflicts.csv").set_index("class")["support"]
    order = sorted(names, key=lambda code: -support[names[code]])

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.38
    for offset, (_, row) in enumerate(weighted.iterrows()):
        change = [row[f"mean_test_f1_class_{code}"] - baseline[f"mean_test_f1_class_{code}"]
                  for code in order]
        ax.bar(np.arange(len(order)) + (offset - 0.5) * width, change, width=width * 0.92,
               color=WEIGHT_COLOURS[row["weights"]], label=_weight_label(row["weights"]))

    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(order)), [names[code] for code in order], rotation=35, ha="right")
    ax.set_ylabel("F1 change vs no weighting")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    _save(fig, output_dir, "fig_tuning_weights_per_class.png")


def tuning_leaf_size(output_dir):
    """
    Q3: macro F1 against the minimum leaf size, one line per impurity measure.
    """

    stage = pd.read_csv(output_dir / "tuning_stages.csv")
    stage = stage[stage["stage"] == 2]
    leaves = sorted(stage["param_classifier__min_samples_leaf"].unique())

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for colour, (criterion, group) in zip(SERIES, stage.groupby("param_classifier__criterion")):
        group = group.set_index("param_classifier__min_samples_leaf").loc[leaves]
        ax.errorbar(range(len(leaves)), group["mean_test_f1_macro"], yerr=group["std_test_f1_macro"],
                    color=colour, marker="o", capsize=3, label=criterion)

    ax.set_xticks(range(len(leaves)), [str(int(v)) for v in leaves])
    ax.set_xlabel("minimum instances per leaf (1 = fully grown trees)")
    ax.set_ylabel("macro F1 (inner CV)")
    ax.legend(title="impurity measure")
    _save(fig, output_dir, "fig_tuning_leaf_size.png")


def tree_count(output_dir):
    """
    Q1: macro F1 against forest size, on the tuning sample and on the full data.
    """

    selected = json.loads((output_dir / "selected_configuration.json").read_text())
    inner = pd.read_csv(output_dir / "tuning_tree_count.csv")
    outer = pd.read_csv(output_dir / "outer_tree_count.csv")
    outer = outer[outer["model"] == "RF"]
    summary = pd.read_csv(output_dir / "cv_summary.csv")
    reference = summary[summary["metric"] == "f1_macro"].set_index("model")["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
    for ax, curve, title in [(axes[0], inner, "tuning sample (inner CV)"),
                             (axes[1], outer, "full dataset (outer CV)")]:
        stats = curve.groupby("trees")["f1_macro"].agg(["mean", "std"])
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                        color=SERIES[2], alpha=0.2, linewidth=0)
        ax.plot(stats.index, stats["mean"], color=SERIES[2], marker="o", label="random forest")
        ax.set_xscale("log")
        ax.set_xlabel("number of trees")
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("macro F1")
    axes[0].axvline(selected["classifier__n_estimators"], color=INK_SECONDARY,
                    linestyle="--", linewidth=1)
    axes[0].annotate(f"chosen: {selected['classifier__n_estimators']}",
                     (selected["classifier__n_estimators"], axes[0].get_ylim()[0]),
                     xytext=(4, 6), textcoords="offset points", fontsize=8, color=INK)

    for key in ["Tree", "kNN"]:
        if key in reference:
            axes[1].axhline(reference[key], color=MODEL_COLOURS[key], linestyle="--", linewidth=1.5)
            axes[1].annotate(MODEL_NAMES[key], (outer["trees"].max(), reference[key]),
                             xytext=(0, 3), textcoords="offset points", ha="right",
                             fontsize=8, color=INK)
    _save(fig, output_dir, "fig_tree_count.png")


# --- evaluation ----------- #


def per_class_f1(output_dir, keys=("kNN", "Tree", "RF")):
    per_class = pd.read_csv(output_dir / "per_class.csv")
    ceiling = pd.read_csv(output_dir / "label_conflicts.csv").set_index("class")["ceiling_f1"]
    order = _class_order(per_class)
    width = 0.8 / len(keys)

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    for offset, key in enumerate(keys):
        frame = per_class[per_class["model"] == key].set_index("class").loc[order]
        ax.bar(np.arange(len(order)) + (offset - (len(keys) - 1) / 2) * width, frame["f1_mean"],
               width=width * 0.92, yerr=frame["f1_std"], capsize=2, error_kw={"linewidth": 0.8},
               color=MODEL_COLOURS[key], label=MODEL_NAMES[key])

    ax.scatter(range(len(order)), ceiling.loc[order], marker="_", s=420, color=INK,
               linewidths=2, zorder=4, label="idealised ceiling")
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    ax.grid(axis="x", visible=False)
    _save(fig, output_dir, "fig_per_class_f1.png")


def confusion_matrices(output_dir, keys=("kNN", "Tree", "RF")):
    matrices = pd.read_csv(output_dir / "confusion_matrices.csv")
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.8),
                             sharey=True, layout="constrained")

    for position, (ax, key) in enumerate(zip(axes, keys)):
        frame = matrices[matrices["model"] == key].drop(columns="model").set_index("actual")
        names = list(frame.index)
        matrix = frame.to_numpy(dtype=float)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)

        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.grid(False)
        if position == 0:
            ax.set_ylabel("actual")
        ax.set_title(MODEL_NAMES[key], fontsize=9, color=INK)
        for i in range(len(names)):
            for j in range(len(names)):
                if matrix[i, j] >= 0.02:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if matrix[i, j] > 0.5 else INK)

    fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02, label="share of actual class")
    _save(fig, output_dir, "fig_confusion_matrices.png")


def fold_scores(output_dir):
    """
    Macro F1 of every model on every outer fold; the grey lines join the same
    fold across models, since the comparison is paired.
    """

    folds = pd.read_csv(output_dir / "cv_folds.csv")
    keys = [k for k in ["kNN", "Tree", "RF-untuned", "RF"] if k in set(folds["model"])]
    table = folds.pivot(index="fold", columns="model", values="f1_macro")[keys]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for _, row in table.iterrows():
        ax.plot(range(len(keys)), row.to_numpy(), color=GRID, linewidth=1, zorder=1)
    for position, key in enumerate(keys):
        ax.scatter(np.full(len(table), position), table[key],
                   color=MODEL_COLOURS.get(key, INK_SECONDARY), s=36, zorder=3)
        ax.annotate(f"{table[key].mean():.4f}", (position, table[key].max()), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)

    ax.set_xticks(range(len(keys)), [MODEL_NAMES[k].replace(", ", "\n").replace(" (", "\n(")
                                     for k in keys])
    ax.set_ylabel("macro F1 per outer fold")
    ax.grid(axis="x", visible=False)
    _save(fig, output_dir, "fig_fold_scores.png")


def bias_variance(output_dir):
    """
    Q5: loss, bias and variance of the tree and the two forests, over all rows
    and averaged over classes (bars: mean over outer folds, error bars: s.d.).
    """

    table = pd.read_csv(output_dir / "bias_variance.csv")
    keys = [k for k in ["Tree", "RF-untuned", "RF"] if k in set(table["model"])]
    parts = ["loss", "bias", "variance"]
    width = 0.8 / len(parts)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True, layout="constrained")
    for ax, scope in zip(axes, ["all rows", "class average"]):
        stats = table[table["scope"] == scope].groupby("model")[parts].agg(["mean", "std"])
        for offset, part in enumerate(parts):
            ax.bar(np.arange(len(keys)) + (offset - 1) * width, stats.loc[keys, (part, "mean")],
                   width=width * 0.92, yerr=stats.loc[keys, (part, "std")], capsize=2,
                   error_kw={"linewidth": 0.8}, color=DECOMPOSITION_COLOURS[part], label=part)
        ax.set_xticks(range(len(keys)), [MODEL_NAMES[k].replace(", ", "\n").replace(" (", "\n(")
                                         for k in keys])
        ax.set_title(f"{scope} ({'like accuracy' if scope == 'all rows' else 'like macro F1'})",
                     fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of predictions")
    axes[0].legend(loc="upper right")
    _save(fig, output_dir, "fig_bias_variance.png")


def strength_correlation(output_dir):
    """
    Q6: Breiman's strength and correlation, and the resulting macro F1, by the
    number of features considered at each split.
    """

    table = pd.read_csv(output_dir / "strength_correlation.csv", keep_default_na=False)
    order = list(dict.fromkeys(table["max_features"]))
    stats = table.groupby("max_features").agg(["mean", "std"]).loc[order]
    x = range(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), layout="constrained")
    for ax, column, label in [(axes[0], "strength", "strength (mean margin)"),
                              (axes[1], "correlation", "correlation between trees")]:
        ax.errorbar(x, stats[(column, "mean")], yerr=stats[(column, "std")], color=SERIES[2],
                    marker="o", capsize=3)
        ax.set_ylabel(label)

    axes[2].errorbar(x, stats[("forest_f1_macro", "mean")], yerr=stats[("forest_f1_macro", "std")],
                     color=SERIES[2], marker="o", capsize=3, label="forest")
    axes[2].errorbar(x, stats[("tree_f1_macro", "mean")], yerr=stats[("tree_f1_macro", "std")],
                     color=INK_SECONDARY, marker="o", linestyle="--", capsize=3,
                     label="single tree (mean)")
    axes[2].set_ylabel("macro F1")
    axes[2].legend(loc="lower right")
    
    for ax in axes:
        ax.set_xticks(x, [_feature_label(v) for v in order], fontsize=7)
        ax.set_xlabel("features considered at each split")
    _save(fig, output_dir, "fig_strength_correlation.png")


def plot_all(output_dir):
    label_conflicts(output_dir)
    tuning_features_and_weights(output_dir)
    tuning_weights_per_class(output_dir)
    tuning_leaf_size(output_dir)
    tree_count(output_dir)
    per_class_f1(output_dir)
    confusion_matrices(output_dir)
    fold_scores(output_dir)
    bias_variance(output_dir)
    strength_correlation(output_dir)
