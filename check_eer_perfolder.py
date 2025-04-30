import sys
import os
import pandas as pd
import eval_metric as em
import re

'''
python eval_mandiri_LA_perfolder.py "/path/to/score_folder" "/path/to/cm_protocols"
python eval_mandiri_LA_perfolder.py "/path/to/score_folder" "/path/to/cm_protocols" "/path/to/output_folder"
python eval_mandiri_LA_perfolder.py /media/dl-1/Second\ Drive/Experiment/Rafid/SSL_Anti-spoofing/score_indo/[folder] /media/dl-1/Second\ Drive/Experiment/Rafid/Dataset/LA/ASVspoof_LA_cm_protocols eval
'''

if len(sys.argv) != 4:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]  # Path to the folder containing files
truth_dir = sys.argv[2]    # Path to the truth folder
phase = sys.argv[3]

# Convert relative paths to absolute paths
submit_file = os.path.abspath(submit_file)
truth_dir = os.path.abspath(truth_dir)

print("Evaluating in folder:", submit_file)

# Paths to CM protocol files
cm_key_file = os.path.join(truth_dir, "ASVspoof2019.LA.cm.eval.trl.txt")

def rename_single_digit_epochs(folder_path):
    """
    Rename files with single-digit epoch numbers to have leading zeros.
    """
    for filename in os.listdir(folder_path):
        # Match pattern 'score_epoch_X.txt' where X is a single digit
        match = re.match(r'(score_epoch_)(\d)(\.txt)', filename)
        if match:
            new_filename = f"{match.group(1)}0{match.group(2)}{match.group(3)}"
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")

def performance(cm_scores, invert=False):
    """
    Compute EER for given CM scores.
    """
    bona_cm = cm_scores[cm_scores[4] == "bonafide"]["1_x"].values
    spoof_cm = cm_scores[cm_scores[4] == "spoof"]["1_x"].values
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0] if not invert else em.compute_eer(-bona_cm, -spoof_cm)[0]
    return eer_cm

def eval_to_score_file(score_file, cm_key_file):
    """
    Evaluate the submission file for CM scores and compute EER.
    """
    cm_data = pd.read_csv(cm_key_file, sep=" ", header=None)
    submission_scores = pd.read_csv(score_file, sep=" ", header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        print(f"CHECK: submission has {len(submission_scores)} of {len(cm_data)} expected trials.")
        exit(1)

    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=1, how="inner")
    eer_cm = performance(cm_scores)
    print(f"EER: {eer_cm}")

    # Check for inverted scores
    eer_cm2 = performance(cm_scores, invert=True)
    print(f"EER2 (inverted): {eer_cm2}")

    return eer_cm, eer_cm2

def evaluate_and_save_to_excel(model_files, cm_key_file, output_folder):
    """
    Evaluate all models and save the EERs to an Excel file.
    """
    eer_results = []

    for model_file in model_files:
        model_path = os.path.join(submit_file, model_file)

        if not os.path.isfile(model_path):
            print(f"File {model_path} not found!")
            continue

        print(f"Evaluating {model_file}...")
        eer, eer2 = eval_to_score_file(model_path, cm_key_file)
        eer_results.append([model_file, eer, eer2])

    if eer_results:
        df = pd.DataFrame(eer_results, columns=["Epoch", "EER", "EER2"])
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "eer_results.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    if not os.path.isdir(submit_file):
        print(f"Directory {submit_file} does not exist.")
        exit(1)

    if not os.path.isdir(truth_dir):
        print(f"Directory {truth_dir} does not exist.")
        exit(1)

    if phase not in ["progress", "eval", "hidden_track"]:
        print("Phase must be either progress, eval, or hidden_track")
        exit(1)

    print(sys.argv)

    # Rename files with single-digit epoch numbers
    rename_single_digit_epochs(submit_file)

    # Directory output for saving evaluation results
    output_folder = submit_file
    model_files = [f for f in os.listdir(submit_file) if f.endswith(".txt")]

    if not model_files:
        print("No model files found in the directory.")
        exit(1)

    print(f"Found {len(model_files)} model files. Starting evaluation...")

    evaluate_and_save_to_excel(model_files, cm_key_file, output_folder)
