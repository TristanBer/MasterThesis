import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def bootstrap_metrics_ci(
    y_true,
    y_pred,
    class_names=None,
    n_bootstraps=10000,
    ci=95,
    average="macro",
    random_state=42,
):
    """
    Compute bootstrap confidence intervals for overall and per-class metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth integer labels from the validation set.
    y_pred : array-like of shape (n_samples,)
        Predicted integer labels from the best-checkpoint model.
    class_names : list of str, optional
        Human-readable class names for per-class reporting.
    n_bootstraps : int
        Number of bootstrap resamples (10,000 is a common robust default).
    ci : float
        Confidence level as a percentage (e.g. 95 for a 95% CI).
    average : str
        Averaging scheme for aggregate precision/recall/F1 ('macro' is
        recommended for imbalanced multi-class problems, as it weights
        every class equally regardless of support).
    random_state : int
        Seed for reproducibility of the resampling procedure.

    Returns
    -------
    results : dict
        Point estimates and (lower, upper) CI bounds for each metric.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    alpha = (100 - ci) / 2

    classes = np.unique(y_true)

    # Storage for bootstrap distributions
    boot = {
        "accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
    }
    per_class_f1 = {c: [] for c in classes}

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)  # sample WITH replacement
        yt, yp = y_true[idx], y_pred[idx]

        boot["accuracy"].append(accuracy_score(yt, yp))
        boot["precision_macro"].append(
            precision_score(yt, yp, average=average, labels=classes, zero_division=0)
        )
        boot["recall_macro"].append(
            recall_score(yt, yp, average=average, labels=classes, zero_division=0)
        )
        boot["f1_macro"].append(
            f1_score(yt, yp, average=average, labels=classes, zero_division=0)
        )

        f1_pc = f1_score(yt, yp, average=None, labels=classes, zero_division=0)
        for c, score in zip(classes, f1_pc):
            per_class_f1[c].append(score)

    def summarize(arr):
        arr = np.asarray(arr)
        return {
            "point": float(np.mean(arr)),
            "lower": float(np.percentile(arr, alpha)),
            "upper": float(np.percentile(arr, 100 - alpha)),
        }

    results = {m: summarize(v) for m, v in boot.items()}
    results["per_class_f1"] = {}
    for c in classes:
        name = class_names[c] if class_names is not None else str(c)
        results["per_class_f1"][name] = summarize(per_class_f1[c])

    return results


def print_ci_report(results, ci=95):
    """Pretty-print the bootstrap CI results for console and thesis tables."""
    print(f"\n=== Bootstrap {ci}% Confidence Intervals "
          f"(nonparametric, percentile method) ===")
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        r = results[metric]
        print(f"  {metric:<18}: {r['point']*100:5.2f}% "
              f"[{r['lower']*100:5.2f}%, {r['upper']*100:5.2f}%]")

    print("\n  Per-class F1:")
    for name, r in results["per_class_f1"].items():
        print(f"    {name:<28}: {r['point']*100:5.2f}% "
              f"[{r['lower']*100:5.2f}%, {r['upper']*100:5.2f}%]")


def plot_ci_results(results, model_name="Model", ci=95, save_path=None):
    """
    Produce two publication-ready figures:
      1. Overall metrics (accuracy, precision, recall, F1-macro) with 95% CI error bars.
      2. Per-class F1 scores with 95% CI error bars.

    Parameters
    ----------
    results : dict
        Output of bootstrap_metrics_ci().
    model_name : str
        Used in the figure titles (e.g. "I3D", "Baseline").
    ci : int
        Confidence level, used only for axis/title labelling.
    save_path : str or None
        If provided, saves both figures to disk:
          <save_path>_overall.png and <save_path>_perclass.png
    """

    # ── Figure 1: Overall metrics ────────────────────────────────────────────
    metric_labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision\n(macro)",
        "recall_macro": "Recall\n(macro)",
        "f1_macro": "F1\n(macro)",
    }
    points = [results[m]["point"] * 100 for m in metric_labels]
    lowers = [results[m]["point"] * 100 - results[m]["lower"] * 100 for m in metric_labels]
    uppers = [results[m]["upper"] * 100 - results[m]["point"] * 100 for m in metric_labels]

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metric_labels))
    bars = ax1.bar(x, points, color="#4878CF", width=0.5, zorder=2)
    ax1.errorbar(
        x, points,
        yerr=[lowers, uppers],
        fmt="none", color="black", capsize=6, linewidth=1.5, zorder=3
    )

    # Annotate point values above each bar
    for bar, pt, lo, up in zip(bars, points, lowers, uppers):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + up + 1.5,
            f"{pt:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels.values(), fontsize=10)
    ax1.set_ylabel("Score (%)", fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.set_title(f"{model_name} — Overall Metrics with {ci}% Bootstrap CI", fontsize=12)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        fig1.savefig(f"{save_path}_overall.png", dpi=300)
        print(f"Saved: {save_path}_overall.png")
    plt.show()

    # ── Figure 2: Per-class F1 ───────────────────────────────────────────────
    class_names = list(results["per_class_f1"].keys())
    pc_points = [results["per_class_f1"][c]["point"] * 100 for c in class_names]
    pc_lowers = [results["per_class_f1"][c]["point"] * 100 - results["per_class_f1"][c]["lower"] * 100 for c in class_names]
    pc_uppers = [results["per_class_f1"][c]["upper"] * 100 - results["per_class_f1"][c]["point"] * 100 for c in class_names]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    x2 = np.arange(len(class_names))
    bars2 = ax2.bar(x2, pc_points, color="#6ACC65", width=0.5, zorder=2)
    ax2.errorbar(
        x2, pc_points,
        yerr=[pc_lowers, pc_uppers],
        fmt="none", color="black", capsize=6, linewidth=1.5, zorder=3
    )

    for bar, pt, up in zip(bars2, pc_points, pc_uppers):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + up + 1.5,
            f"{pt:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(class_names, fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("F1 Score (%)", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_title(f"{model_name} — Per-class F1 with {ci}% Bootstrap CI", fontsize=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        fig2.savefig(f"{save_path}_perclass.png", dpi=300)
        print(f"Saved: {save_path}_perclass.png")
    plt.show()