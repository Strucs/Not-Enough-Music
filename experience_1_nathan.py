#
import torch
#
import lib_dataset as ld
#
from matplotlib import pyplot as plt
import numpy as np
#
from lib_model_ast_classification import ASTClassification


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
data: ld.DatasetAudios = ld.DatasetAudios()

#
x_train, y_train = data.get_batch_train(2)

#
print(f"x_train {x_train.shape} y_train {y_train.shape}")
print(f"y_train = {y_train}")

#
# data = torch.transpose(torch.transpose( x_train[0], dim0=1, dim1=2 ), dim0=0, dim1=2).to(torch.int32).numpy()
# plot_rgb_image( data )

#
with torch.no_grad():
    #
    outputs = model(x_train)

#
print(f"outputs = {outputs} | {outputs.shape}")
