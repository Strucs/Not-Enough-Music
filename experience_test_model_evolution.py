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
#
from experience_lib import load_model
#
from lib_device import get_device



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

    if len(sys.argv) >= 3:
        #
        del sys.argv[2]

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
    base_path: str = f"model_weights/{sys.argv[1]}/"
    #
    base_ev: str = f"model_evolutions/{sys.argv[1]}/"

    #
    base_ev_pca: str = f"{base_ev}PCA/"
    base_ev_tsne: str = f"{base_ev}TSNE/"
    base_ev_conf_matr: str = f"{base_ev}conf_matr/"

    #
    if not os.path.exists(base_ev_pca):
        os.makedirs(base_ev_pca)

    #
    if not os.path.exists(base_ev_tsne):
        os.makedirs(base_ev_tsne)

    #
    if not os.path.exists(base_ev_conf_matr):
        os.makedirs(base_ev_conf_matr)

    #
    top_1_acc: dict[int, float] = {}
    top_3_acc: dict[int, float] = {}

    #
    print(f"os.listdir( base_path ) = {os.listdir( base_path )}")

    #
    for model_file_path in os.listdir( base_path ):

        print(f"{model_file_path} ?")

        #
        prefix: str = "weights_"

        #
        if not model_file_path.startswith(prefix): continue
        #
        if not model_file_path.endswith(".pth"): continue

        #
        i: int = model_file_path.find("__")

        #
        if i == -1: continue

        #
        id_epoch: int = int( model_file_path[len(prefix):i] )

        #
        print(f"\nEpoch {id_epoch}\n")


        print(f"Loading model from path `{base_path}{model_file_path}` ...")
        model.load_state_dict( torch.load(f"{base_path}{model_file_path}", map_location="cpu") )
        model = model.to( get_device() )

        #
        top_1_acc[id_epoch] = calculate_top_k_accuracy(dataset=dataset, model=model, k=1, batch_size=1)
        top_3_acc[id_epoch] = calculate_top_k_accuracy(dataset=dataset, model=model, k=3, batch_size=1)

        #
        ee = "0" * ( (4 - len(str(id_epoch))) ) + str(id_epoch)

        calculate_confusion_matrix(dataset=dataset, model=model, batch_size=1, class_names=class_names, save_plot=f"{base_ev_conf_matr}{ee}.png")
        calculate_pca_embeddings(dataset=dataset, model=model, batch_size=1, class_names=class_names, save_plot=f"{base_ev_pca}{ee}.png")
        calculate_tsne_embeddings(dataset=dataset, model=model, batch_size=1, class_names=class_names, save_plot=f"{base_ev_tsne}{ee}.png")

    #
    top_1_acc = dict(sorted(top_1_acc.items(), key=lambda item: item[1]))
    top_3_acc = dict(sorted(top_3_acc.items(), key=lambda item: item[1]))

    #
    plt.clf()
    plt.plot(list(top_1_acc.values()) )
    plt.savefig( f"{base_ev}top_1_acc.png" )

    #
    plt.clf()
    plt.plot( list(top_3_acc.values()) )
    plt.savefig( f"{base_ev}top_3_acc.png" )


#
if __name__ == "__main__":
    #
    main()
