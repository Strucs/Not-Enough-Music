#
import torch
from torch import Tensor
import torch.nn as nn
#
from vit_pytorch import ViT   # type: ignore
from vit_pytorch import SimpleViT   # type: ignore


#
class VitClassifier(nn.Module):
    def __init__(self, num_classes: int, image_size: tuple[int, int] = (192, 320)) -> None:

        super().__init__()

        #
        self.vit = ViT(
            image_size = image_size[-2:],
            patch_size = 64,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # Add classifier layer
        self.class_lin: nn.Linear = nn.Linear(1000, num_classes)

    #
    def get_embedding(self, x: Tensor) -> Tensor:

        #
        return self.vit(x)


    #
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass through the ResNet model.

        Args:
            x (Tensor): The input tensor. Expected shape is (Batch Size, Channels, Height, Width).
                         For typical image data, Channels is 3. For adaptations to 1D data
                         represented as 2D, the number of channels might vary.

        Returns:
            Tensor: The output tensor after passing through the ResNet and the final
                    classification layer. Shape is (Batch Size, num_classes).
        """

        # The input tensor x is passed directly through the modified ResNet model
        #
        x = self.vit(x)
        #
        x = self.class_lin(x)
        #
        return x


#
class SimpleVitClassifier(nn.Module):
    def __init__(self, num_classes: int, image_size: tuple[int, int] = (192, 320)) -> None:

        super().__init__()

        #
        self.vit = SimpleViT(
            image_size = image_size[-2:],
            patch_size = 64,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )

        # Add classifier layer
        self.class_lin: nn.Linear = nn.Linear(1000, num_classes)

    #
    def get_embedding(self, x: Tensor) -> Tensor:

        #
        return self.vit(x)


    #
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass through the ResNet model.

        Args:
            x (Tensor): The input tensor. Expected shape is (Batch Size, Channels, Height, Width).
                         For typical image data, Channels is 3. For adaptations to 1D data
                         represented as 2D, the number of channels might vary.

        Returns:
            Tensor: The output tensor after passing through the ResNet and the final
                    classification layer. Shape is (Batch Size, num_classes).
        """

        # The input tensor x is passed directly through the modified ResNet model
        #
        x = self.vit(x)
        #
        x = self.class_lin(x)
        #
        return x
