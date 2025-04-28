
import torch
from torch import Tensor
import torch.nn as nn


# Définition du modèle CNN 1D
class SimpleCNN1D(nn.Module):

    def __init__(self, input_channels, num_classes) -> None:

        super(SimpleCNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * (675808 // 2 // 2), num_classes)

    #
    def get_embedding(self, x: Tensor) -> Tensor:

        #
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        #
        return x


    def forward(self, x: Tensor) -> Tensor:

        #
        x = self.get_embedding(x)

        #
        x = self.fc(x)

        #
        return x
