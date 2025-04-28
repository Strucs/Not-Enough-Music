import torch
import torch.nn as nn
import torch.optim as optim
import lib_dataset as ld
from lib_training import train_simple_epochs_loop

from model_angel2 import SimpleCNN1D

# Chargement des données
data: ld.DatasetAudios = ld.DatasetAudios()
x_train, y_train = data.get_batch_train(2)

input_channels = 1
num_classes = 10
model = SimpleCNN1D(input_channels, num_classes)

print(f"input: {input_channels}")

loss_fn: nn.Module = nn.CrossEntropyLoss()

train_simple_epochs_loop(
    dataset = data,
    model = model,
    loss_fn = loss_fn,
    optimizer_type = "adam",
    optimizer_kwargs = {
        "params": model.parameters(),
        "lr": 0.001
    },
    nb_epochs = 20,
    batch_size = -1,
    batch_parallel_calcul = 5,
    model_saving_folder = "model_weights/model_legna_2/",
    model_save_epochs_steps = 1
)
