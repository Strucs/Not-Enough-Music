#
from typing import Optional
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
from lib_accuracy import calculate_top_k_accuracy
from lib_accuracy import calculate_confusion_matrix
from lib_accuracy import calculate_pca_embeddings
from lib_accuracy import calculate_tsne_embeddings
from lib_accuracy import calculate_unsupervized_clusters
#
from experience_lib import load_model




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
    class_names: Optional[list[str]] = None

    #
    if hasattr(dataset, "class_names"):
        #
        class_names = dataset.class_names

    #
    print(f"CLASS_NAMES = {class_names}")

    #
    if sys.argv[1].startswith("pre_train_"):
        #
        calculate_unsupervized_clusters(dataset=dataset, model=model, dataset_part="test")
        calculate_unsupervized_clusters(dataset=dataset, model=model, dataset_part="train")
    #
    else:
        #
        calculate_top_k_accuracy(dataset=dataset, model=model, k=1, batch_size=1)
        calculate_top_k_accuracy(dataset=dataset, model=model, k=3, batch_size=1)

        calculate_confusion_matrix(dataset=dataset, model=model, batch_size=1, class_names=class_names)
        calculate_pca_embeddings(dataset=dataset, model=model, batch_size=1, class_names=class_names)
        calculate_tsne_embeddings(dataset=dataset, model=model, batch_size=1, class_names=class_names)


#
if __name__ == "__main__":
    #
    main()
