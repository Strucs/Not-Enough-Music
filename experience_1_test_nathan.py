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
from lib_accuracy import calculate_accuracy, calculate_top_k_accuracy


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
if __name__ == "__main__":

    #
    model_weights_path: str = ""

    #
    if len(sys.argv) > 1:

        #
        model_weights_path = sys.argv[1]

    #
    model: ASTClassification = ASTClassification(nb_classes = 10)

    #
    if model_weights_path != "":

        #
        if not os.path.exists( model_weights_path ):

            #
            raise FileNotFoundError(f"Path doesn't exists : `{model_weights_path}`")

        #
        model.load_state_dict( torch.load(model_weights_path) )

    #
    dataset: ld.DatasetAudios = ld.DatasetAudios()

    #
    # calculate_accuracy(dataset, model)

    calculate_top_k_accuracy(dataset=dataset, model=model, k=1, batch_size=1)
    calculate_top_k_accuracy(dataset=dataset, model=model, k=3, batch_size=1)
