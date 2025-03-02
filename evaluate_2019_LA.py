#!/usr/bin/env python
"""
Script to compute pooled EER and min tDCF for ASVspoof2021 LA. 
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has tje CM protocol and ASV score.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track
Example:
$: python evaluate.py eval_CM_scores_file_SSL_LA.txt ~/Dataset/ASVspoof29019_LA/LA/ eval
"""

import sys, os.path
import numpy as np
import pandas
import eval_metric_LA as em
from glob import glob

if len(sys.argv) != 4:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
truth_dir = sys.argv[2]
phase = sys.argv[3]

# truth_dir = ./keys
asv_key_file = os.path.join(truth_dir, "ASVspoof2019_LA_asv_protocols/")
asv_scr_file = os.path.join(
    truth_dir, "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
)
cm_key_file = os.path.join(
    truth_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
)

## Kode asli
# asv_key_file = os.path.join(truth_dir, 'ASV/trial_metadata.txt')
# asv_scr_file = os.path.join(truth_dir, 'ASV/ASVTorch_Kaldi/score.txt')
# cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')


Pspoof = 0.05
cost_model = {
    "Pspoof": Pspoof,  # Prior probability of a spoofing attack
    "Ptar": (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    "Pnon": (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    "Cmiss": 1,  # Cost of tandem system falsely rejecting target speaker
    "Cfa": 10,  # Cost of tandem system falsely accepting nontarget speaker
    "Cfa_spoof": 10,  # Cost of tandem system falsely accepting spoof
}


def load_asv_metrics():
    # Load ASV scores using numpy like in the second code
    asv_data = np.genfromtxt(asv_scr_file, dtype=str)
    asv_keys = asv_data[:, 1]  # The second column contains the labels
    asv_scores = asv_data[:, 2].astype(float)  # The third column contains the scores

    # Extract target, non-target, and spoof scores as per the second code
    target_asv = asv_scores[asv_keys == "target"]
    nontarget_asv = asv_scores[asv_keys == "nontarget"]
    spoof_asv = asv_scores[asv_keys == "spoof"]

    # Calculate EER and ASV error rates
    eer_asv, asv_threshold = em.compute_eer(target_asv, nontarget_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(
        target_asv, nontarget_asv, spoof_asv, asv_threshold
    )

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[4] == "bonafide"]["1_x"].values
    spoof_cm = cm_scores[cm_scores[4] == "spoof"]["1_x"].values

    if invert == False:
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = em.compute_eer(-bona_cm, -spoof_cm)[0]

    if invert == False:
        tDCF_curve, _ = em.compute_tDCF(
            bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False
        )
    else:
        tDCF_curve, _ = em.compute_tDCF(
            -bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False
        )

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_to_score_file(score_file, cm_key_file):
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics()
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

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=1, how="inner")
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100 * eer_cm)
    print(out_data, end="")

    # just in case that the submitted file reverses the sign of positive and negative scores
    min_tDCF2, eer_cm2 = performance(
        cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=True
    )

    if min_tDCF2 < min_tDCF:
        print(
            "CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking"
            % (min_tDCF, min_tDCF2)
        )

    if min_tDCF == min_tDCF2:
        print(
            "WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?"
        )

    return min_tDCF


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
