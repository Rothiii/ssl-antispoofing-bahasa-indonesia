import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

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


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


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


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()

    fname_list = []
    score_list = []

    with torch.no_grad():  # Menggunakan no_grad untuk menghemat memori
        for batch_x, utt_id in data_loader:
            fname_list = []
            score_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)

            batch_out = model(batch_x)

            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

            with open(save_path, "a+") as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write("{} {}\n".format(f, cm))
            fh.close()
    print("Scores saved to {}".format(save_path))


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
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")
    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        default="E:/Data Kuliah/Tugas Akhir (Skripsi)/Dataset DIY/LA/",
        help="Change this to user's full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.",
    )
    """
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    """

    parser.add_argument(
        "--protocols_path",
        type=str,
        default="E:/Data Kuliah/Tugas Akhir (Skripsi)/Dataset DIY/LA/",
        help="Change with path to user's LA database protocols directory adisdress",
    )
    """
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
    """

    # Hyperparameters
    parser.add_argument(
        "--models_folder",
        type=str,
        help="Folder yang berisi model-model yang sudah dilatih",
    )

    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default="weighted_CCE")
    # model
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed (default: 1234)"
    )

    parser.add_argument("--model_path", type=str, default=None, help="Model checkpoint")
    parser.add_argument(
        "--comment", type=str, default=None, help="Comment to describe the saved model"
    )
    # Auxiliary arguments
    parser.add_argument(
        "--track", type=str, default="LA", choices=["LA", "PA", "DF"], help="LA/PA/DF"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path to save the evaluation result",
    )
    parser.add_argument("--eval", action="store_true", default=False, help="eval mode")
    parser.add_argument(
        "--is_eval", action="store_true", default=False, help="eval database"
    )
    parser.add_argument("--eval_part", type=int, default=0)
    # backend options
    parser.add_argument(
        "--cudnn-deterministic-toggle",
        action="store_false",
        default=True,
        help="use cudnn-deterministic? (default true)",
    )

    parser.add_argument(
        "--cudnn-benchmark-toggle",
        action="store_true",
        default=False,
        help="use cudnn-benchmark? (default false)",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument(
        "--algo",
        type=int,
        default=5,
        help="Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]",
    )

    # LnL_convolutive_noise parameters
    parser.add_argument(
        "--nBands",
        type=int,
        default=5,
        help="number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]",
    )
    parser.add_argument(
        "--minF",
        type=int,
        default=20,
        help="minimum centre frequency [Hz] of notch filter.[default=20] ",
    )
    parser.add_argument(
        "--maxF",
        type=int,
        default=8000,
        help="maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]",
    )
    parser.add_argument(
        "--minBW",
        type=int,
        default=100,
        help="minimum width [Hz] of filter.[default=100] ",
    )
    parser.add_argument(
        "--maxBW",
        type=int,
        default=1000,
        help="maximum width [Hz] of filter.[default=1000] ",
    )
    parser.add_argument(
        "--minCoeff",
        type=int,
        default=10,
        help="minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]",
    )
    parser.add_argument(
        "--maxCoeff",
        type=int,
        default=100,
        help="maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]",
    )
    parser.add_argument(
        "--minG",
        type=int,
        default=0,
        help="minimum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--maxG",
        type=int,
        default=0,
        help="maximum gain factor of linear component.[default=0]",
    )
    parser.add_argument(
        "--minBiasLinNonLin",
        type=int,
        default=5,
        help=" minimum gain difference between linear and non-linear components.[default=5]",
    )
    parser.add_argument(
        "--maxBiasLinNonLin",
        type=int,
        default=20,
        help=" maximum gain difference between linear and non-linear components.[default=20]",
    )
    parser.add_argument(
        "--N_f",
        type=int,
        default=5,
        help="order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]",
    )

    # ISD_additive_noise parameters
    parser.add_argument(
        "--P",
        type=int,
        default=10,
        help="Maximum number of uniformly distributed samples in [%].[defaul=10]",
    )
    parser.add_argument(
        "--g_sd", type=int, default=2, help="gain parameters > 0. [default=2]"
    )

    # SSI_additive_noise parameters
    parser.add_argument(
        "--SNRmin",
        type=int,
        default=10,
        help="Minimum SNR value for coloured additive noise.[defaul=10]",
    )
    parser.add_argument(
        "--SNRmax",
        type=int,
        default=40,
        help="Maximum SNR value for coloured additive noise.[defaul=40]",
    )

    ##===================================================Rawboost data augmentation ======================================================================#

    if not os.path.exists("models"):
        os.mkdir("models")
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ["LA", "PA", "DF"], "Invalid track given"

    # database
    prefix = "ASVspoof2019_{}".format(track)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    # define model saving path
    model_tag = "model_{}_{}_{}_{}_{}_{}".format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr, args.algo
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
    model = model.to(device)
    print("nb_params:", nb_params)

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print("Model loaded : {}".format(args.model_path))
        else:
            print("Checkpoint does not contain expected keys. Starting from scratch.")


    # evaluation
    # #base is 2021 but I change to 2019
    if args.eval:
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
        num_workers=4,
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
    recent_checkpoints = []

    for epoch in range(start_epoch, num_epochs):
        print("\nTrainig will Start for epoch: ", epoch)
        running_loss = train_epoch(train_loader, model, args.lr, optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)

        # Log ke tensorboard
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("loss", running_loss, epoch)

        # Print loss pada setiap epoch
        print("\n epoch[{}] - Running Loss[{}] - Val Loss[{}] ".format(epoch, running_loss, val_loss))

        # Simpan model
        checkpoint_path = os.path.join(model_save_path, "epoch_{}.pth".format(epoch))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            },
            checkpoint_path,
        )

        # Update recent checkpoints
        recent_checkpoints.append(checkpoint_path)
        if len(recent_checkpoints) > 2:
            # Remove the oldest checkpoint
            os.remove(recent_checkpoints.pop(0))

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_save_path, "best.pth")
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                },
                best_model_path,
            )
