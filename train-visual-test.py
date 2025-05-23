import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import re
from check_eer import eval_to_score_file

# import yaml
from data_utils_SSL import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
)
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import pandas as pd
from args_config import get_args
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from scipy.stats import norm


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def get_latest_epoch(models_folder):
    models_folder = os.path.abspath(models_folder)+"/"  # Ubah jadi absolute path

    if not os.path.exists(models_folder):
        print(f"Error: Directory {models_folder} does not exist!")
        return 0, None

    model_files = [f for f in os.listdir(models_folder) if f.endswith(".pth")]
    if not model_files:
        return 0, None

    # Extract epoch numbers from filenames
    epochs = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in model_files if re.search(r'epoch_(\d+)', f)]
    if not epochs:
        return 0, None

    latest_epoch = max(epochs)
    latest_model_path = os.path.join(models_folder, f"epoch_{latest_epoch}.pth")
    return latest_epoch, latest_model_path


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():  # Menggunakan no_grad untuk menghemat memori
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)

            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size

    val_loss /= num_total

    return val_loss

import matplotlib.pyplot as plt

# def visualize_features(features, utt_id, step):
#     """Visualize spectral features of the input audio"""
#     plt.figure(figsize=(10, 4))
#     plt.imshow(features.cpu().numpy()[0], aspect='auto', origin='lower')
#     plt.colorbar()
#     plt.title(f"Spectral Features for {utt_id[0]} (Step {step})")
#     plt.xlabel("Time")
#     plt.ylabel("Frequency")
#     plt.savefig(f"debug_visualizations/spectral_features_{step}.png")
#     plt.close()
#     print(f"Spectral features visualization saved to debug_visualizations/spectral_features_{step}.png")

def visualize_features(features, utt_id, step):
    print(f"Feature shape: {features[0].shape}")  # atau batch_x.shape
    feature = features[0]
    if feature.ndim == 3:  # Misal (channel, freq, time)
        feature = feature[0]  # ambil channel pertama
    elif feature.ndim == 4:  # (batch, channel, freq, time), ini aneh kalau udah index 0
        feature = feature[0][0]
    plt.figure(figsize=(10, 4))
    plt.imshow(feature.cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f"Spectral Features for {utt_id[0]} (Step {step})")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, threshold=0):
    """
    Plot confusion matrix for spoofing detection results.
    y_true: Ground truth labels (0 for spoof, 1 for bonafide)
    y_pred: Raw scores from model
    threshold: Classification threshold (default 0)
    """
    # Convert scores to binary predictions based on threshold
    binary_preds = [1 if score > threshold else 0 for score in y_pred]
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Bonafide'])
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix for ASV Spoofing Detection')
    plt.savefig('debug_visualizations/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to debug_visualizations/confusion_matrix.png")
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    
    return accuracy, far, frr

def plot_score_distribution(scores, labels):
    """
    Plot the distribution of scores for both spoof and bonafide samples.
    """
    plt.figure(figsize=(12, 6))
    
    # Separate scores for spoof and bonafide
    spoof_scores = [s for s, l in zip(scores, labels) if l == 0]
    bonafide_scores = [s for s, l in zip(scores, labels) if l == 1]
    
    # Plot histograms
    plt.hist(spoof_scores, bins=50, alpha=0.7, label='Spoof', color='red')
    plt.hist(bonafide_scores, bins=50, alpha=0.7, label='Bonafide', color='green')
    
    plt.axvline(x=0, color='black', linestyle='--', label='Threshold (0)')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Scores for Spoof and Bonafide Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('debug_visualizations/score_distribution.png')
    plt.close()
    print("Score distribution saved to debug_visualizations/score_distribution.png")

def plot_roc_curve(y_true, scores):
    """
    Plot ROC curve to visualize model performance.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('debug_visualizations/roc_curve.png')
    plt.close()
    print("ROC curve saved to debug_visualizations/roc_curve.png")
    
    # Calculate EER
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diffs)
    eer = (fpr[min_idx] + fnr[min_idx]) / 2
    print(f"Equal Error Rate (EER): {eer:.4f}")
    
    return eer

def plot_det_curve(y_true, scores):
    """
    Plot Detection Error Tradeoff (DET) curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    
    # Filter out extreme values to avoid infinity in ppf
    valid_indices = np.logical_and(fpr > 0.001, fpr < 0.999) & np.logical_and(fnr > 0.001, fnr < 0.999)
    fpr_filtered = fpr[valid_indices]
    fnr_filtered = fnr[valid_indices]
    
    plt.figure(figsize=(10, 8))
    
    # Convert to normal deviate space
    fpr_norm = norm.ppf(fpr_filtered)
    fnr_norm = norm.ppf(fnr_filtered)
    
    plt.plot(fpr_norm, fnr_norm, 'b', label='DET curve')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('Detection Error Tradeoff (DET)')
    
    # Calculate and plot EER point
    abs_diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diffs)
    eer = (fpr[min_idx] + fnr[min_idx]) / 2
    
    if fpr[min_idx] > 0.001 and fpr[min_idx] < 0.999 and fnr[min_idx] > 0.001 and fnr[min_idx] < 0.999:
        plt.plot(norm.ppf(fpr[min_idx]), norm.ppf(fnr[min_idx]), 'ro', label=f'EER = {eer:.4f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('debug_visualizations/det_curve.png')
    plt.close()
    print("DET curve saved to debug_visualizations/det_curve.png")
    
    return eer

def visualize_layer_activations(model, input_batch, device):
    """
    Visualize activations of intermediate layers to understand feature extraction
    Note: This function requires model modification to work properly
    """
    original_input = input_batch.clone()
    model.eval()
    
    # Forward pass with stored activations (requires model modifications)
    with torch.no_grad():
        activations = {}
        hooks = []
        
        # Create hook function to store activations
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for layers you want to visualize
        # Example - modify based on your model architecture
        if hasattr(model, 'backbone'):
            # For SSL model with backbone
            hooks.append(model.backbone.register_forward_hook(hook_fn('backbone')))
            if hasattr(model, 'proj'):
                hooks.append(model.proj.register_forward_hook(hook_fn('projection')))
        
        # For generic CNN models
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if len(hooks) < 5:  # Limit to first few layers
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        output = model(input_batch.to(device))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Plot activations
        plt.figure(figsize=(15, 10))
        
        # Plot original input
        plt.subplot(2, 3, 1)
        plt.imshow(original_input.cpu().numpy()[0], aspect='auto', origin='lower')
        plt.title("Original Input")
        
        # Plot activations from layers
        idx = 2
        for name, activation in activations.items():
            plt.subplot(2, 3, idx)
            
            # Handle different activation shapes
            if len(activation.shape) == 4:  # Conv output: B x C x H x W
                # Take first channel of first sample and average across channels
                act_vis = activation[0].mean(dim=0).cpu().numpy()
            elif len(activation.shape) == 3:  # Some feature maps: B x F x T
                act_vis = activation[0].mean(dim=0).cpu().numpy()
            else:  # Linear layer or other: B x F
                act_vis = activation[0].cpu().numpy()
                act_vis = act_vis.reshape(1, -1)
            
            plt.imshow(act_vis, aspect='auto')
            plt.title(f"{name[:15]} Activation")
            idx += 1
            if idx > 6:  # Show only 5 activations + input
                break
        
        plt.tight_layout()
        plt.savefig('debug_visualizations/layer_activations.png')
        plt.close()
        print("Layer activations saved to debug_visualizations/layer_activations.png")

def produce_evaluation_file(dataset, model, device, save_path):
    """
    Perform evaluation and generate visualization of the classification process
    - Shows spectral features
    - Prints detailed model outputs
    - Generates confusion matrix, ROC curve, DET curve and score distributions
    - Visualizes internal model activations
    """
    # Create directory for debug visualizations
    os.makedirs("debug_visualizations", exist_ok=True)
    
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()

    all_scores = []
    all_labels = []
    all_utt_ids = []
    
    print("\n===== STARTING DETAILED EVALUATION WITH VISUALIZATION =====\n")
    with torch.no_grad():
        for i, (batch_x, utt_id) in enumerate(data_loader):
            print(f"\n----- STEP {i + 1}: Processing batch with {len(utt_id)} samples -----")
            
            # Step 1: Visualize spectral features of input audio
            if i < 5:  # Limit to first 5 batches to avoid too many plots
                visualize_features(batch_x, utt_id, i + 1)
            
            # Step 2: Send inputs to model
            batch_x = batch_x.to(device)
            print(f"Input features shape: {batch_x.shape}")
            
            # Step 3: Visualize internal model activations (for selected batches)
            if i == 0:
                try:
                    visualize_layer_activations(model, batch_x[:1], device)
                except Exception as e:
                    print(f"Could not visualize layer activations: {str(e)}")
            
            # Step 4: Get model prediction
            batch_out = model(batch_x)
            print(f"Model output shape: {batch_out.shape}")
            print(f"Raw logits: {batch_out}")
            
            # Step 5: Get probabilities
            probs = torch.softmax(batch_out, dim=1)
            print(f"Softmax probabilities: {probs}")
            
            # Step 6: Extract scores (higher means more likely to be bonafide)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            print(f"Final scores: {batch_score}")
            
            # Step 7: Interpretation of scores for each sample
            for idx, score in enumerate(batch_score):
                prediction = "BONAFIDE" if score > 0 else "SPOOF"
                strength = abs(score)
                confidence_level = ""
                if strength > 4:
                    confidence_level = "Very Strong"
                elif strength > 2:
                    confidence_level = "Strong"
                elif strength > 1:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Weak"
                    
                print(f"  Sample {utt_id[idx]}: Score {score:.4f} → {prediction} ({confidence_level} confidence: {strength:.4f})")
            
            # Storage for later visualization
            all_scores.extend(batch_score)
            all_utt_ids.extend(utt_id)
            
            # Save results to file
            with open(save_path, "a+") as fh:
                for f, cm in zip(utt_id, batch_score):
                    fh.write(f"{f} {cm}\n")
    
    print("\n----- EVALUATION RESULTS -----")
    print(f"Processed {len(all_scores)} samples")
    print(f"Score range: Min={min(all_scores):.4f}, Max={max(all_scores):.4f}")
    print(f"Average score: {sum(all_scores)/len(all_scores):.4f}")
    
    # Calculate distribution statistics
    bonafide_pred = [s for s in all_scores if s > 0]
    spoof_pred = [s for s in all_scores if s <= 0]
    print(f"Predicted bonafide: {len(bonafide_pred)} samples")
    print(f"Predicted spoof: {len(spoof_pred)} samples")
    
    # Generate additional visualizations if we have ground truth
    if hasattr(dataset, 'labels'):
        print("\n----- GENERATING PERFORMANCE VISUALIZATIONS -----")
        # Get ground truth labels
        labels = []
        for id in all_utt_ids:
            if id in dataset.labels:
                labels.append(dataset.labels[id])
            else:
                # If we can't find the label, make a guess based on score
                idx = all_utt_ids.index(id)
                labels.append(1 if all_scores[idx] > 0 else 0)
        
        all_labels = labels
        
        # Calculate performance metrics and generate visualizations
        plot_score_distribution(all_scores, all_labels)
        plot_confusion_matrix(all_labels, all_scores)
        eer = plot_roc_curve(all_labels, all_scores)
        plot_det_curve(all_labels, all_scores)
        
        print(f"Equal Error Rate (EER): {eer:.4f}")
    else:
        print("\nNote: No ground truth labels available for performance visualization.")
        print("To generate confusion matrix, ROC and DET curves, use a dataset with labels.")
    
    print("\n===== EVALUATION COMPLETED =====")
    print(f"Scores saved to: {save_path}")
    print(f"Visualizations saved to: debug_visualizations/ folder")
    
    return all_scores, all_utt_ids

def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0

    num_total = 0.0

    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in train_loader:

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        batch_loss = criterion(batch_out, batch_y)

        running_loss += batch_loss.item() * batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total

    return running_loss


if __name__ == "__main__":
    # Import arguments from args_config.py
    args = get_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = "LA"

    # database
    prefix = "ASVspoof2019_{}".format(track)
    prefix_2019 = "ASVspoof2019.{}".format(track)

    # define model saving path
    model_tag = "{}_{}_{}_{}_{}".format(
        args.loss, args.num_epochs, args.batch_size, args.lr, args.algo
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_save_path = os.path.join("models", model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    # ! Berbeda dengan main ssl df
    model = model.to(device)
    # ! --------------------------
    print("Number of parameters in the model: ", nb_params)

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch, latest_model_path = get_latest_epoch(model_save_path)
    if latest_model_path:
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint)\

        print("Model loaded from epoch {}: {}".format(start_epoch - 1, latest_model_path))
    else:
        if args.model_path:
            start_epoch, _ = get_latest_epoch(os.path.dirname(args.model_path))
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print('Model loaded from: {}'.format(args.model_path))
        else:
            print("No checkpoint found. Starting from scratch.")


    # evaluation
    if args.eval and args.models_folder:
        model_files = [f for f in os.listdir(args.models_folder) if f.endswith(".pth")]

        if not model_files:
            print("Tidak ada model yang ditemukan di folder tersebut.")
            sys.exit(1)

        for model_file in model_files:
            model_path = os.path.join(args.models_folder, model_file)
            print(f"Evaluating model: {model_path}")

            # Membuat model baru dan memuat state dict-nya
            model = Model(args, device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Evaluasi hanya jika --eval diaktifkan
            if args.eval:
                torch.cuda.empty_cache()  # Kosongkan cache CUDA sebelum memulai evaluasi

                # Membaca file evaluasi
                file_eval = genSpoof_list(
                    dir_meta=os.path.join(
                        args.protocols_path,
                        "{}_cm_protocols/{}.cm.eval.trl.txt".format(
                            prefix, prefix_2019
                        ),
                    ),
                    is_train=False,
                    is_eval=True,
                )
                print("Jumlah evaluasi trial:", len(file_eval))

                eval_set = Dataset_ASVspoof2021_eval(
                    list_IDs=file_eval,
                    base_dir=os.path.join(
                        args.database_path + "ASVspoof2019_{}_eval/".format(args.track)
                    ),
                )

                # Menentukan path untuk file output evaluasi
                output_filename = f"score_{model_file.replace('.pth', '.txt')}"
                eval_output_path = os.path.join(args.eval_output, output_filename)

                # Melakukan evaluasi dan menyimpan hasil
                produce_evaluation_file(eval_set, model, device, eval_output_path)

                print(
                    f"Evaluasi selesai untuk model: {model_file}, hasil disimpan di {eval_output_path}"
                )
        sys.exit(0)
    
    if args.eval and args.model_path:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2019)),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        eval_to_score_file(args.eval_output, "/media/dl-1/Second Drive/Experiment/Rafid/Dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
        sys.exit(0)

    # define train dataloader
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path
            + "{}_cm_protocols/{}.cm.train.trn.txt".format(prefix, prefix_2019)
        ),
        is_train=True,
        is_eval=False,
    )

    print("no. of training trials", len(file_train))

    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(
            args.database_path
            + "{}_{}_train/".format(prefix_2019.split(".")[0], args.track)
        ),
        algo=args.algo,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    del train_set, d_label_trn

    # define validation dataloader

    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path
            + "{}_cm_protocols/{}.cm.dev.trl.txt".format(prefix, prefix_2019)
        ),
        is_train=False,
        is_eval=False,
    )

    print("no. of validation trials", len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(
            args.database_path
            + "{}_{}_dev/".format(prefix_2019.split(".")[0], args.track)
        ),
        algo=args.algo,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, num_workers=8, shuffle=False
    )
    del dev_set, d_label_dev
    print("Data loaded")

    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter("logs/{}".format(model_tag))

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        print("\nTraining will Start for epoch: ", epoch)
        running_loss = train_epoch(train_loader, model, args.lr, optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)

        # Log ke tensorboard
        writer.add_scalar("Eval loss", val_loss, epoch)
        writer.add_scalar("Running loss", running_loss, epoch)

        # Print loss pada setiap epoch
        print("\n epoch[{}] - Running Loss[{}] - Val Loss[{}] ".format(epoch, running_loss, val_loss))

        # Simpan model
        if epoch % 1 == 0:
            checkpoint_path = os.path.join(model_save_path, "epoch_{}.pth".format(epoch))
            torch.save( model.state_dict(), checkpoint_path )
