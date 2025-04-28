#
import sys
import os
#
import torch
from torch import nn
#
import lib_dataset as ld
#
from matplotlib import pyplot as plt
import numpy as np
#
from lib_training import train_simple_epochs_loop
#
from experience_lib import load_model



#
def main() -> None:

    #
    if len(sys.argv) == 1:
        #
        raise UserWarning("please indicate the class type you want to train")

    #
    model: nn.Module
    #
    dataset: ld.Dataset
    #
    model_saving_folder: str = f"model_weights/{sys.argv[1]}/"

    #
    model, dataset = load_model()

    #
    if model is None or dataset is None:
        #
        raise UserWarning(f"model or dataset not correctly loaded...\nModel = {model}\nDataset = {dataset}")

    #
    loss_fn: nn.Module = nn.CrossEntropyLoss()

    #
    train_simple_epochs_loop(
        dataset = dataset,
        model = model,
        loss_fn = loss_fn,
        optimizer_type = "adam",
        optimizer_kwargs = {
            "params": model.parameters(),
            "lr": 0.001
        },
        nb_epochs = 20,
        batch_size = -1,
        batch_parallel_calcul = 1,
        model_saving_folder = model_saving_folder,
        model_save_epochs_steps = 1
    )

#
if __name__ == "__main__":
    #
    main()
