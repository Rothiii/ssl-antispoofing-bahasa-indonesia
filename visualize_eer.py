#!/usr/bin/env python

"""
Script to visualize EER (Equal Error Rate) for ASVspoof scores.
This script generates visualizations to help understand the score distributions
and the relationship between False Acceptance Rate and False Rejection Rate.

Usage:
$: python visualize_eer.py PATH_TO_SCORE_FILE PATH_TO_GROUNDTRUTH_DIR

 -PATH_TO_SCORE_FILE: path to the score file (e.g., score_epoch_00.txt)
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol.

Example:
$: python visualize_eer.py scores/adjust-lr/score_epoch_00.txt dataset/LA/
"""

import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eval_metric as em
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def parse_scores(score_file):
    """
    Parse score file into a pandas DataFrame
    """
    try:
        scores_df = pd.read_csv(
            score_file, sep=" ", header=None, names=["file_id", "score"]
        )
        print(f"Successfully loaded {len(scores_df)} scores from {score_file}")
        return scores_df
    except Exception as e:
        print(f"Error parsing score file: {e}")
        sys.exit(1)


def load_ground_truth(protocol_file):
    """
    Load ground truth labels from protocol file
    """
    try:
        # Read the protocol file
        cm_data = pd.read_csv(protocol_file, sep=" ", header=None)
        # Create a dictionary mapping file IDs to labels
        labels_dict = dict(zip(cm_data[1], cm_data[4]))
        print(f"Successfully loaded {len(labels_dict)} labels from {protocol_file}")
        return labels_dict, cm_data
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        sys.exit(1)


def merge_scores_with_labels(scores_df, cm_data):
    """
    Merge scores with ground truth labels
    """
    # Merge the scores with the ground truth data
    merged_data = scores_df.merge(cm_data, left_on="file_id", right_on=1, how="inner")
    print(f"Successfully merged {len(merged_data)} entries")
    return merged_data


def calculate_and_visualize_eer(merged_data, output_dir="./eer_visualizations"):
    """
    Calculate EER and generate visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract scores for bonafide and spoof samples
    bonafide_scores = merged_data[merged_data[4] == "bonafide"]["score"].values
    spoof_scores = merged_data[merged_data[4] == "spoof"]["score"].values

    # Calculate EER using the existing eval_metric module
    eer, eer_threshold = em.compute_eer(bonafide_scores, spoof_scores)

    print(f"\nEqual Error Rate (EER): {eer:.6f}")
    print(f"Threshold at EER: {eer_threshold:.6f}")

    # Get FRR and FAR for different thresholds
    frr, far, thresholds = em.compute_det_curve(bonafide_scores, spoof_scores)

    # Find the index where the threshold is closest to the EER threshold
    threshold_idx = np.argmin(np.abs(thresholds - eer_threshold))

    # Calculate how many legitimate users are rejected and impostors accepted at EER threshold
    legitimate_rejected = np.sum(bonafide_scores < eer_threshold)
    impostors_accepted = np.sum(spoof_scores >= eer_threshold)

    total_legitimate = len(bonafide_scores)
    total_impostors = len(spoof_scores)

    print(f"\nAt EER threshold ({eer_threshold:.6f}):")
    print(
        f"- Legitimate users rejected: {legitimate_rejected}/{total_legitimate} ({legitimate_rejected/total_legitimate:.2%})"
    )
    print(
        f"- Impostors accepted: {impostors_accepted}/{total_impostors} ({impostors_accepted/total_impostors:.2%})"
    )

    # Calculate ROC curve for plotting
    y_true = np.concatenate(
        [np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))]
    )
    y_scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 1. Plot DET Curve (FAR vs. FRR)
    plt.figure(figsize=(10, 8))
    plt.plot(far, frr, "b-", linewidth=2, label="DET Curve")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Random Classifier")
    plt.scatter(
        far[threshold_idx],
        frr[threshold_idx],
        marker="o",
        color="red",
        s=100,
        label=f"EER = {eer:.4f} at threshold = {eer_threshold:.4f}",
    )

    # Add text box with statistics
    stats_text = (
        f"Equal Error Rate (EER): {eer:.4f}\n"
        f"Threshold at EER: {eer_threshold:.4f}\n\n"
        f"At EER threshold:\n"
        f"- Legitimate users rejected: {legitimate_rejected}/{total_legitimate} ({legitimate_rejected/total_legitimate:.2%})\n"
        f"- Impostors accepted: {impostors_accepted}/{total_impostors} ({impostors_accepted/total_impostors:.2%})"
    )

    plt.annotate(
        stats_text,
        xy=(0.5, 0.25),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    plt.xlabel("False Acceptance Rate (FAR)", fontsize=12)
    plt.ylabel("False Rejection Rate (FRR)", fontsize=12)
    plt.title("Detection Error Tradeoff (DET) Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    det_curve_path = os.path.join(output_dir, "det_curve.png")
    plt.savefig(det_curve_path)
    print(f"DET curve saved to {det_curve_path}")

    # 2. Plot Score Distributions
    plt.figure(figsize=(10, 6))
    bins = np.linspace(
        min(np.min(bonafide_scores), np.min(spoof_scores)),
        max(np.max(bonafide_scores), np.max(spoof_scores)),
        100,
    )

    plt.hist(
        bonafide_scores,
        bins=bins,
        alpha=0.7,
        label="Bonafide",
        color="green",
        density=True,
    )
    plt.hist(
        spoof_scores, bins=bins, alpha=0.7, label="Spoof", color="red", density=True
    )
    plt.axvline(
        x=eer_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"EER Threshold = {eer_threshold:.4f}",
    )
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("Normalized Count (Density)", fontsize=12)
    plt.title("Score Distribution", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    score_dist_path = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(score_dist_path)
    print(f"Score distribution saved to {score_dist_path}")

    # 3. Plot ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    print(f"ROC curve saved to {roc_curve_path}")

    # 4. Plot threshold effect
    plt.figure(figsize=(12, 6))
    threshold_range = np.linspace(np.min(thresholds), np.max(thresholds), 1000)
    frr_interp = np.interp(threshold_range, thresholds, frr)
    far_interp = np.interp(threshold_range, thresholds, far)

    plt.plot(
        threshold_range,
        frr_interp,
        "g-",
        linewidth=2,
        label="False Rejection Rate (FRR)",
    )
    plt.plot(
        threshold_range,
        far_interp,
        "r-",
        linewidth=2,
        label="False Acceptance Rate (FAR)",
    )
    plt.axvline(
        x=eer_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"EER Threshold = {eer_threshold:.4f}",
    )
    plt.axhline(
        y=eer, color="blue", linestyle="--", linewidth=2, label=f"EER = {eer:.4f}"
    )

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.title("Error Rates vs. Threshold", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    threshold_effect_path = os.path.join(output_dir, "threshold_effect.png")
    plt.savefig(threshold_effect_path)
    print(f"Threshold effect plot saved to {threshold_effect_path}")
    # 5. Create confusion matrix
    # Create binary predictions using the EER threshold
    bonafide_pred = (bonafide_scores >= eer_threshold).astype(
        int
    )  # 1 if above threshold (predicted bonafide)
    spoof_pred = (spoof_scores >= eer_threshold).astype(
        int
    )  # 1 if above threshold (predicted bonafide)

    # True labels (1 for bonafide, 0 for spoof)
    bonafide_true = np.ones_like(bonafide_pred)
    spoof_true = np.zeros_like(spoof_pred)

    # Combine for the full confusion matrix
    y_true = np.concatenate([bonafide_true, spoof_true])
    y_pred = np.concatenate([bonafide_pred, spoof_pred])

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )  # 1 is bonafide (positive class)
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )  # Sensitivity/recall for bonafide
    f1 = f1_score(y_true, y_pred, zero_division=0)

    specificity = (
        cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    )  # TN/(TN+FP)

    # True positive, false negative, false positive, true negative
    tn, fp = cm[0, 0], cm[0, 1]  # For spoof samples
    fn, tp = cm[1, 0], cm[1, 1]  # For bonafide samples

    # Replace the confusion matrix plotting section with this improved version
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))  # Increased figure size for better layout

    # Create a gridspec for better layout management
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])  

    # Plot the confusion matrix on the left
    ax0 = plt.subplot(gs[0])
    im = ax0.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax0.set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Set labels
    classes = ["Spoof", "Bonafide"]
    tick_marks = np.arange(len(classes))
    ax0.set_xticks(tick_marks)
    ax0.set_xticklabels(classes, fontsize=12)
    ax0.set_yticks(tick_marks)
    ax0.set_yticklabels(classes, fontsize=12)
    ax0.set_xlabel("Predicted Label", fontsize=12)
    ax0.set_ylabel("True Label", fontsize=12)

    # Add text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax0.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    # Add metrics information as text in the right subplot
    ax1 = plt.subplot(gs[1])
    ax1.axis('off')  # Turn off the axes for the text area

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall/Sensitivity: {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"F1 Score: {f1:.4f}\n\n"
        f"True Positive: {tp} (Bonafide correctly identified)\n"
        f"False Negative: {fn} (Bonafide misclassified as Spoof)\n"
        f"False Positive: {fp} (Spoof misclassified as Bonafide)\n"
        f"True Negative: {tn} (Spoof correctly identified)"
    )

    ax1.text(
        0, 0.5, metrics_text,
        va="center", ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    )

    plt.tight_layout(pad=2.0)  # Add padding between subplots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")

    # Save numerical data for further analysis
    results = {
        "EER": eer,
        "EER_threshold": eer_threshold,
        "Bonafide_rejected": legitimate_rejected,
        "Bonafide_total": total_legitimate,
        "Spoof_accepted": impostors_accepted,
        "Spoof_total": total_impostors,
        "FRR_at_EER": frr[threshold_idx],
        "FAR_at_EER": far[threshold_idx],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_Score": f1,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
    }

    output_file = os.path.join(output_dir, "eval_result.xlsx")
    pd.DataFrame.from_dict(results, orient="index", columns=["Value"]).to_excel(output_file)

    print(f"Vertical Excel saved to {output_file}")

    return results


def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("ERROR: invalid input arguments. Please read the instruction below:")
        print(__doc__)
        sys.exit(1)

    score_file = sys.argv[1]
    truth_dir = sys.argv[2]

    # Check if files exist
    if not os.path.isfile(score_file):
        print(f"ERROR: Score file {score_file} doesn't exist")
        sys.exit(1)

    if not os.path.isdir(truth_dir):
        print(f"ERROR: Truth directory {truth_dir} doesn't exist")
        sys.exit(1)

    # Path to the CM protocol file
    cm_key_file = os.path.join(truth_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

    if not os.path.isfile(cm_key_file):
        print(f"ERROR: Protocol file {cm_key_file} not found in the truth directory")
        sys.exit(1)

    # Parse score file
    scores_df = parse_scores(score_file)

    # Load ground truth labels
    labels_dict, cm_data = load_ground_truth(cm_key_file)

    # Merge scores with labels
    merged_data = merge_scores_with_labels(scores_df, cm_data)

    # Create output directory based on score file name
    score_basename = os.path.basename(score_file)
    output_dir = os.path.join("eer_visualizations", os.path.splitext(score_basename)[0])

    # Calculate EER and generate visualizations
    results = calculate_and_visualize_eer(merged_data, output_dir)

    print("\nEER Visualization complete!")
    print(f"All visualizations saved in {output_dir}/")

    # Also check with inverted scores
    print("\nChecking with inverted scores...")
    bonafide_scores = merged_data[merged_data[4] == "bonafide"]["score"].values
    spoof_scores = merged_data[merged_data[4] == "spoof"]["score"].values

    eer = em.compute_eer(bonafide_scores, spoof_scores)[0]
    eer_inverted = em.compute_eer(-bonafide_scores, -spoof_scores)[0]

    if eer_inverted < eer:
        print(f"WARNING: Inverting scores gives a lower EER!")
        print(f"Original EER: {eer:.6f}, Inverted EER: {eer_inverted:.6f}")
        print(
            "This suggests that your class labels might be swapped or the model output might be inverted."
        )

        # Calculate EER with inverted scores and generate visualizations
        merged_data_inverted = merged_data.copy()
        merged_data_inverted["score"] = -merged_data_inverted["score"]
        inverted_output_dir = os.path.join(output_dir, "inverted_scores")
        calculate_and_visualize_eer(merged_data_inverted, inverted_output_dir)

        print(f"Inverted score visualizations saved in {inverted_output_dir}/")


if __name__ == "__main__":
    main()
