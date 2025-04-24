import torch
import torch.nn as nn
import torch.optim as optim
import lib_dataset as ld
from lib_training import train_simple_epochs_loop

# Chargement des données
data: ld.DatasetAudios = ld.DatasetAudios()
x_train, y_train = data.get_batch_train(2)

# Définition du modèle CNN 1D
class SimpleCNN1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * (x_train.shape[1] // 2 // 2), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    model_saving_folder = "model_weights/model_legna_1/",
    model_save_epochs_steps = 1
)
