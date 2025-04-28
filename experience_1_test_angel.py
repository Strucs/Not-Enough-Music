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
from lib_model_ast_classification import ASTClassification
#
from lib_training import train_simple_epochs_loop
from lib_accuracy import calculate_accuracy

from model_angel1 import SimpleCNN1D

# Chargement des données
data: ld.DatasetAudios = ld.DatasetAudios()
x_train, y_train = data.get_batch_train(2)

#
if __name__ == "__main__":

    #
    model_weights_path: str = ""

    #
    if len(sys.argv) > 1:

        #
        model_weights_path = sys.argv[1]

    #
    if not os.path.exists( model_weights_path ):

        #
        raise FileNotFoundError(f"Path doesn't exists : `{model_weights_path}`")


    input_channels = 1
    num_classes = 10
    model = SimpleCNN1D(input_channels, num_classes)

    #
    if model_weights_path != "":

        #
        model.load_state_dict( torch.load(model_weights_path) )

    #
    dataset: ld.DatasetAudios = ld.DatasetAudios()

    #
    calculate_accuracy(dataset, model)
