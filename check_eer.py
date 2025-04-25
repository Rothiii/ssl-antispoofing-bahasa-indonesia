#!/usr/bin/env python

"""
Script to compute EER for ASVspoof2021 LA without ASV.
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUNDTRUTH_DIR phase

 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol.
 -phase: either progress, eval, or hidden_track
Example:
$: python eval_mandiri_LA.py eval_CM_scores_file_SSL_LA.txt ~/Dataset/LA/ eval
"""

import sys, os.path
import numpy as np
import pandas
import eval_metric as em

if len(sys.argv) != 4:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
truth_dir = sys.argv[2]
phase = sys.argv[3]

# Paths to CM protocol files
cm_key_file = os.path.join(
    truth_dir, "ASVspoof2019.LA.cm.eval.trl.txt"
)


def performance(cm_scores, invert=False):
    """
    Compute EER for given CM scores.
    """
    bona_cm = cm_scores[cm_scores[4] == "bonafide"]["1_x"].values
    spoof_cm = cm_scores[cm_scores[4] == "spoof"]["1_x"].values
    if invert == False:
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = em.compute_eer(-bona_cm, -spoof_cm)[0]

    return eer_cm


def eval_to_score_file(score_file, cm_key_file):
    """
    Evaluate the submission file for CM scores and compute EER.
    """
    cm_data = pandas.read_csv(cm_key_file, sep=" ", header=None)
    submission_scores = pandas.read_csv(
        score_file, sep=" ", header=None, skipinitialspace=True
    )

    if len(submission_scores) != len(cm_data):
        print(
            "CHECK: submission has %d of %d expected trials."
            % (len(submission_scores), len(cm_data))
        )
        exit(1)

    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=1, how="inner")
    eer_cm = performance(cm_scores)
    out_data = "eer: %.2f\n" % (100 * eer_cm)
    print(out_data, end="")

    # Check for inverted scores
    eer_cm2 = performance(cm_scores, invert=True)

    if eer_cm2 < eer_cm:
        print(
            "CHECK: we negated your scores and achieved a lower EER. Before: %.3f - Negated: %.3f - your class labels might be swapped."
            % (eer_cm, eer_cm2)
        )

    if eer_cm == eer_cm2:
        print(
            "WARNING: Negating your scores gives the same EER. Ensure your classifier is working correctly."
        )

    return eer_cm


if __name__ == "__main__":

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)

    if not os.path.isdir(truth_dir):
        print("%s doesn't exist" % (truth_dir))
        exit(1)

    if phase != "progress" and phase != "eval" and phase != "hidden_track":
        print("phase must be either progress, eval, or hidden_track")
        exit(1)

    _ = eval_to_score_file(submit_file, cm_key_file)
