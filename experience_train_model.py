#
import sys
import os
import random
#
import torch
from torch import Tensor
from torch import nn
#
import lib_dataset as ld
#
from matplotlib import pyplot as plt
import numpy as np
#
from lib_training import train_simple_epochs_loop
#
from experience_lib import load_model, load_dataset
#
import lib_loss as ll




# Random Seed at file level
random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)


#
def main() -> None:

    #
    if len(sys.argv) == 1:
        #
        raise UserWarning("please indicate the class type you want to train")

    #
    model_saving_folder: str = f"model_weights/{sys.argv[1]}/"

    #
    model: nn.Module = load_model(model_name = sys.argv[1], model_weights_path = sys.argv[2] if len(sys.argv) > 2 else "")
    #
    dataset: ld.Dataset = load_dataset(model_name = sys.argv[1], give_train_dataset=False)

    #
    loss_fn: nn.Module = nn.CrossEntropyLoss()

    loss_fn = ll.FocalLoss(gamma = 2)

    #
    if sys.argv[1].startswith("pre_train_"):

        #
        for j in range(30):

            #
            print(f"\n\nTRAINING PART {j}\n\n")

            #
            crt_dataset: ld.Dataset = load_dataset(model_name = sys.argv[1], give_train_dataset=True, from_dataset = dataset)

            #
            train_simple_epochs_loop(
                dataset = crt_dataset,
                model = model,
                loss_fn = loss_fn,
                optimizer_type = "adam",
                optimizer_kwargs = {
                    "params": model.parameters(),
                    "lr": 0.001
                },
                nb_epochs = 2,
                batch_size = -1,
                batch_parallel_calcul = 1,
                model_saving_folder = model_saving_folder,
                model_save_epochs_steps = 1
            )

            #
            del crt_dataset

    else:

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
            nb_epochs = 100,
            batch_size = -1,
            batch_parallel_calcul = 1,
            model_saving_folder = model_saving_folder,
            model_save_epochs_steps = 1
        )

#
if __name__ == "__main__":
    #
    main()
