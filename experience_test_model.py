#
from typing import Optional
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
from lib_accuracy import calculate_top_k_accuracy
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
    model, dataset = load_model()

    #
    if model is None or dataset is None:
        #
        raise UserWarning(f"model or dataset not correctly loaded...\nModel = {model}\nDataset = {dataset}")

    #
    calculate_top_k_accuracy(dataset=dataset, model=model, k=1, batch_size=1)
    calculate_top_k_accuracy(dataset=dataset, model=model, k=3, batch_size=1)



#
if __name__ == "__main__":
    #
    main()
