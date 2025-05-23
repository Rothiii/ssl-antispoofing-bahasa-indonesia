import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import re
import importlib

# import yaml
from data_utils_SSL import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
)
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import pandas as pd
from args_config import get_args


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
    # Import arguments from args_config.py
    args = get_args()

    if args.sa:
        model_module = importlib.import_module("model")
    else:
        model_module = importlib.import_module("model_without_sa")

    Model = getattr(model_module, "Model")

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

    if args.sa:
        model_tag = model_tag + "_SA"    
    model_save_path = os.path.join("models", model_tag)

    # set model save directory
    if not (args.is_eval and args.eval):
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

        print("Model loaded from epoch {}: {}".format(start_epoch, latest_model_path))
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
