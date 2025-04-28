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
from lib_device import get_device

#
def plot_rgb_image(image_array, title="RGB Image"):
    """
    Plots an RGB image represented as a NumPy array.

    Args:
        image_array: A NumPy array of shape (height, width, 3) representing the RGB image.
        title: Title of the plot.
    """

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Image array must have shape (height, width, 3) for RGB.")

    plt.imshow(image_array)
    plt.title(title)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()



#
def load_model() -> tuple[nn.Module, ld.Dataset]:


    #
    if sys.argv[1] == "AST_classif_1":

        #
        from lib_model_ast_classification import ASTClassification

        #
        dataset = ld.DatasetAudios()
        model = ASTClassification(nb_classes = 10)

    #
    elif sys.argv[1] == "small_conv_1":

        #
        from model_angel1 import SimpleCNN1D

        #
        dataset = ld.DatasetAudios()

        model = SimpleCNN1D(input_channels = len(dataset.get_batch_train(1)[0]), num_classes = 10)  # type: ignore

        #
        print("aa")

    #
    elif sys.argv[1] == "small_conv_2":

        #
        from model_angel2 import SimpleCNN1D  # type: ignore

        #
        dataset = ld.DatasetAudios()
        model = SimpleCNN1D(input_channels = len(dataset.get_batch_train(1)[0]), num_classes = 10)  # type: ignore

    #
    elif sys.argv[1].startswith("resnet"):

        #
        from model_resnet import Resnet

        #
        dataset = ld.DatasetImages()  # type: ignore
        model = Resnet(image_size = dataset.get_batch_train(1)[0].shape, num_classes = 10, resnet_version = sys.argv[1], pretrained= True)  # type: ignore

    #
    else:

        #
        raise UserWarning(f"Unknown model class : `{sys.argv[1]}`")


    #
    if model is None or dataset is None:
        #
        raise UserWarning(f"model or dataset not correctly loaded...\nModel = {model}\nDataset = {dataset}")

    #
    model_weights_path: str = ""

    #
    if len(sys.argv) > 2:

        #
        model_weights_path = sys.argv[2]

    #
    if model_weights_path != "":

        #
        if not os.path.exists( model_weights_path ):

            #
            raise FileNotFoundError(f"Path doesn't exists : `{model_weights_path}`")

        #
        print(f"Loading model from path `{model_weights_path}` ...")
        model.load_state_dict( torch.load(model_weights_path, map_location=get_device()) )


    #
    return (model, dataset)
