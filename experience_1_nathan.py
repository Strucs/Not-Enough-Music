#
import torch
from transformers import AutoProcessor, ASTModel
#
import lib_dataset as ld
#
from matplotlib import pyplot as plt
import numpy as np


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
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", use_fast=True)
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

#
# data.sampling_rate, return_tensors="pt")
pre_inputs = [ processor(x_train[i], sampling_rate=16000, return_tensors="pt") for i in range(len(x_train)) ]

#
inputs = torch.zeros( (len(pre_inputs), 1024, 128) )

#
for i, inp in enumerate(pre_inputs):

    #
    inputs[i] = inp["input_values"][0]

#
with torch.no_grad():
    #
    outputs = model(inputs)

#
last_hidden_states = outputs.last_hidden_state

#
print(f"last_hidden_states = {last_hidden_states} | {last_hidden_states.shape}")
