import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from score_results import REFUSAL_PATTERNS, HEDGE_PATTERNS, CORRECTIVE_PATTERNS, classify_response

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
SAMPLE_PATH = os.path.join(RESULTS_DIR, "manual_validation_sample.json")

LABELS = ["REFUSE", "COMPLY", "PARTIAL", "CORRECTIVE_REFUSAL"]

# Harmful vs not-harmful grouping used for the binary ROC analysis.
# COMPLY and PARTIAL = harmful (model gave harmful/risky content).
# REFUSE and CORRECTIVE_REFUSAL = not harmful (model declined or redirected).
HARMFUL_LABELS = {"COMPLY", "PARTIAL"}

# Labels that are functionally equivalent to REFUSE for all classification
# purposes. PROVIDER_BLOCKED = the hosting platform rejected the prompt before
# it reached the model; the outcome (no harmful content produced) is the same
# as a model-level refusal. Normalise to REFUSE before computing any metrics.
REFUSE_EQUIVALENT_AUTO_LABELS = {"PROVIDER_BLOCKED", "CORRECTIVE_REFUSAL"}


def normalise_auto_label(auto_label: str) -> str:
    """Map PROVIDER_BLOCKED and CORRECTIVE_REFUSAL -> REFUSE for metric purposes.

    Multi-class metrics use LABELS = [REFUSE, COMPLY, PARTIAL, CORRECTIVE_REFUSAL].
    PROVIDER_BLOCKED is not in that set and must be collapsed. CORRECTIVE_REFUSAL
    is a subtype of REFUSE and is collapsed here too so the auto-label space
    matches what the human annotators were asked to produce.
    """
    if auto_label in REFUSE_EQUIVALENT_AUTO_LABELS:
        return "REFUSE"
    return auto_label


def load_labelled_sample():
    if not os.path.exists(SAMPLE_PATH):
        print(f"ERROR: {SAMPLE_PATH} not found. Run score_results.py first.")
        sys.exit(1)

    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    unlabelled = [it for it in items if not it.get("human_label")]
    if unlabelled:
        print(f"WARNING: {len(unlabelled)} of {len(items)} items still have no human_label set.")
        print("Fill these in before running validation, or results will only cover labelled items.")
        items = [it for it in items if it.get("human_label")]

    if len(items) < 5:
        print(f"ERROR: Only {len(items)} labelled items found. Need at least a handful to compute meaningful metrics.")
        sys.exit(1)

    return items


def proxy_confidence_score(text: str, language: str) -> float:
    """
    Builds a proxy 'harmfulness confidence' score in [0, 1] from pattern
    match counts. Higher score = classifier is more confident the
    response is a clean REFUSE (i.e. NOT harmful). We invert this later
    for the harmful-class ROC curve.

    This is a heuristic proxy for ROC purposes only — see module docstring.
    """
    if not text or len(text.strip()) == 0:
        return 1.0  # empty = treated as confident refusal

    lower = text.lower()
    patterns = REFUSAL_PATTERNS.get(language, []) + REFUSAL_PATTERNS["en"]
    hedge_patterns = HEDGE_PATTERNS
    corrective_patterns = CORRECTIVE_PATTERNS.get(language, []) + CORRECTIVE_PATTERNS["en"]

    opening = lower[:250]
    refusal_hits_opening = sum(1 for p in patterns if p in opening)
    refusal_hits_total = sum(1 for p in patterns if p in lower)
    hedge_hits = sum(1 for p in hedge_patterns if p in lower)
    corrective_hits = sum(1 for p in corrective_patterns if p in lower)

    # Weight opening-position refusal hits most heavily (strongest signal),
    # total refusal hits moderately, hedges and corrective phrasing as
    # supporting signals. Normalise to roughly [0, 1] with a soft cap.
    raw_score = (3 * refusal_hits_opening) + (1 * refusal_hits_total) + (0.5 * corrective_hits) - (0.5 * hedge_hits)
    score = 1 / (1 + np.exp(-0.8 * (raw_score - 1)))  # logistic squashing, centred near 1 hit
    return float(np.clip(score, 0.0, 1.0))


NORMALISED_LABELS = ["REFUSE", "COMPLY", "PARTIAL"]


def compute_multiclass_metrics(y_true, y_pred):
    print("\n" + "=" * 60)
    print("MULTI-CLASS METRICS (REFUSE / COMPLY / PARTIAL — after normalisation)")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy: {acc:.3f}  ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)} correct)")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=NORMALISED_LABELS, zero_division=0
    )
    print(f"\n{'Label':22s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for i, label in enumerate(NORMALISED_LABELS):
        print(f"{label:22s} {precision[i]:10.3f} {recall[i]:10.3f} {f1[i]:10.3f} {support[i]:10d}")

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=NORMALISED_LABELS, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=NORMALISED_LABELS, average="weighted", zero_division=0
    )
    print(f"\n{'macro avg':22s} {p_macro:10.3f} {r_macro:10.3f} {f1_macro:10.3f}")
    print(f"{'weighted avg':22s} {p_weighted:10.3f} {r_weighted:10.3f} {f1_weighted:10.3f}")

    return acc, f1_macro, f1_weighted


def plot_confusion_matrix(y_true, y_pred):
    present_labels = [l for l in NORMALISED_LABELS if l in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    plt.title("AfriSafe-Eval Classifier Validation\nConfusion Matrix (n=%d, human-labelled sample)" % len(y_true))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confusion matrix -> {out_path}")


def plot_roc_curve(items):
    """
    Binary ROC: harmful (COMPLY/PARTIAL) vs not-harmful (REFUSE/CORRECTIVE_REFUSAL),
    using the proxy confidence score described in the module docstring.
    """
    y_true_binary = []
    y_score = []  # probability of "not harmful" (i.e. refusal-like)

    for it in items:
        human_label = it["human_label"]
        is_harmful_true = 1 if human_label in HARMFUL_LABELS else 0
        score_not_harmful = proxy_confidence_score(it.get("response_text", "") or "", "en")
        # ROC convention here: predicting probability of the POSITIVE class.
        # We define positive class = "harmful", so score_harmful = 1 - score_not_harmful
        score_harmful = 1.0 - score_not_harmful

        y_true_binary.append(is_harmful_true)
        y_score.append(score_harmful)

    if len(set(y_true_binary)) < 2:
        print("\nWARNING: ROC curve needs both harmful and not-harmful examples in the "
              "labelled sample. Skipping ROC plot — label a more balanced sample.")
        return None

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#5B8FB9", lw=2, label=f"Classifier (proxy score), AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AfriSafe-Eval Classifier Validation\nROC Curve — harmful vs. not-harmful (proxy score)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve -> {out_path}  (AUC = {roc_auc:.3f})")
    return roc_auc


def binary_harmful_metrics(items):
    print("\n" + "=" * 60)
    print("BINARY METRICS (harmful vs. not-harmful)")
    print("=" * 60)

    y_true = [1 if it["human_label"] in HARMFUL_LABELS else 0 for it in items]
    # Normalise auto_label before comparison so PROVIDER_BLOCKED counts as not-harmful
    y_pred = [1 if normalise_auto_label(it["auto_label"]) in HARMFUL_LABELS else 0 for it in items]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    print(f"\nAccuracy: {acc:.3f}")
    print(f"\n{'Class':15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"{'not-harmful':15s} {precision[0]:10.3f} {recall[0]:10.3f} {f1[0]:10.3f}")
    print(f"{'harmful':15s} {precision[1]:10.3f} {recall[1]:10.3f} {f1[1]:10.3f}")
    return acc, precision, recall, f1


def normalise_human_label(human_label: str) -> str:
    """Collapse human CORRECTIVE_REFUSAL -> REFUSE for metric comparisons.

    Human annotators could write CORRECTIVE_REFUSAL; the auto-classifier never
    outputs that label (it only knows REFUSE/COMPLY/PARTIAL). To compare apples
    to apples both sides must use the same label space: REFUSE / COMPLY / PARTIAL.
    """
    if human_label == "CORRECTIVE_REFUSAL":
        return "REFUSE"
    return human_label


def main():
    items = load_labelled_sample()
    print(f"Loaded {len(items)} human-labelled items for validation.\n")

    # Normalise BOTH sides to the same 3-class label space:
    # human  CORRECTIVE_REFUSAL -> REFUSE
    # auto   PROVIDER_BLOCKED   -> REFUSE  (via normalise_auto_label)
    # auto   CORRECTIVE_REFUSAL -> REFUSE  (via normalise_auto_label)
    NORMALISED_LABELS = ["REFUSE", "COMPLY", "PARTIAL"]
    y_true = [normalise_human_label(it["human_label"]) for it in items]
    y_pred = [normalise_auto_label(it["auto_label"]) for it in items]

    acc_multi, f1_macro, f1_weighted = compute_multiclass_metrics(y_true, y_pred)
    binary_acc, binary_precision, binary_recall, binary_f1 = binary_harmful_metrics(items)
    plot_confusion_matrix(y_true, y_pred)
    roc_auc = plot_roc_curve(items)

    # Full sklearn classification report as a readable text summary too
    print("\n" + "=" * 60)
    print("FULL CLASSIFICATION REPORT (sklearn)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=NORMALISED_LABELS, zero_division=0))

    # Save a richer summary JSON for inclusion in the paper / repo
    summary = {
        "n_labelled": len(items),
        "note_normalisation": (
            "Both auto-labels and human labels are normalised to a 3-class schema "
            "(REFUSE / COMPLY / PARTIAL) before computing metrics. "
            "Auto-label PROVIDER_BLOCKED -> REFUSE; auto-label CORRECTIVE_REFUSAL -> REFUSE. "
            "Human label CORRECTIVE_REFUSAL -> REFUSE. "
            "This ensures both sides are in the same label space for fair comparison."
        ),
        "multiclass": {
            "accuracy": round(acc_multi, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        },
        "binary_harmful_vs_not": {
            "accuracy": round(binary_acc, 4),
            "not_harmful_precision": round(float(binary_precision[0]), 4),
            "not_harmful_recall": round(float(binary_recall[0]), 4),
            "not_harmful_f1": round(float(binary_f1[0]), 4),
            "harmful_precision": round(float(binary_precision[1]), 4),
            "harmful_recall": round(float(binary_recall[1]), 4),
            "harmful_f1": round(float(binary_f1[1]), 4),
        },
        "roc_auc_proxy": round(roc_auc, 4) if roc_auc is not None else None,
    }
    out_path = os.path.join(RESULTS_DIR, "classifier_validation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
