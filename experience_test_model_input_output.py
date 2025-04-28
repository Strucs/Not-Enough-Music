#
from typing import Optional
#
import sys
import os
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
from lib_accuracy import calculate_top_k_accuracy
from lib_accuracy import calculate_confusion_matrix
from lib_accuracy import calculate_pca_embeddings
from lib_accuracy import calculate_tsne_embeddings
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
    x: Tensor
    y: Tensor
    x, y = dataset.get_batch_test(1)
    print(f"model input shape : {x.shape}")
    print(f"y shape : {y.shape}")
    print(f"model embedding shape : {model.get_embedding(x).shape}")    # type: ignore
    print(f"model output shape : {model(x).shape}")

#
if __name__ == "__main__":
    #
    main()
