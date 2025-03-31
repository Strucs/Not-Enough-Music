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
model: ASTClassification = ASTClassification(nb_classes = 10)

#
dataset: ld.DatasetAudios = ld.DatasetAudios()

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
    model_saving_folder = "model_weights/model_nathan_1/",
    model_save_at_epochs = 1
)
