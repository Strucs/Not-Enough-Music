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
