#
import torch
from torch import Tensor
from torch import nn


#
class FeedForward(nn.Module):

    #
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dtype: torch.dtype = torch.float32) -> None:

        #
        super().__init__()

        #
        self.lin1: nn.Linear = nn.Linear(in_features=in_dim, out_features=hidden_dim, dtype=dtype)
        self.activ: nn.ReLU = nn.ReLU()
        self.lin2: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=out_dim, dtype=dtype)

    #
    def forward(self, X: Tensor) -> Tensor:

        #
        X = self.lin1(X)
        X = self.activ(X)
        X = self.lin2(X)

        #
        return X


#
class BilinearFeedForward(nn.Module):

    #
    def __init__(self, in1_dim: int, in2_dim: int, hidden_dim: int, out_dim: int, dtype: torch.dtype = torch.float32) -> None:

        #
        super().__init__()

        #
        self.lin1: nn.Bilinear = nn.Bilinear(in1_features=in1_dim, in2_features=in2_dim, out_features=hidden_dim)
        self.activ: nn.ReLU = nn.ReLU()
        self.lin2: nn.Bilinear = nn.Bilinear(in1_features=hidden_dim, in2_features=in2_dim, out_features=out_dim)

    #
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:

        #
        X = self.lin1(X, Y)
        X = self.activ(X)
        X = self.lin2(X, Y)

        #
        return X

