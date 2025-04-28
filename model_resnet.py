import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models  # type: ignore


#
class Resnet(nn.Module):
    def __init__(self, image_size: tuple[int, int], num_classes: int, resnet_version: str = "resnet18", pretrained: bool = True) -> None:
        """
        Initializes a ResNet model adapted for classification.

        Args:
            image_size (tuple[int, int]): The expected input image size (height, width).
                                          Note: Standard torchvision ResNets can handle
                                          variable input sizes, this parameter is mostly
                                          for context or potential future use like input validation.
            num_classes (int): The number of output classes for the classification layer.
            resnet_version (str): The version of the ResNet model to use (e.g., "resnet18",
                                  "resnet34", "resnet50", etc.). Must be available in torchvision.models.
            pretrained (bool): If True, loads pre-trained weights from ImageNet.
        """
        super().__init__()

        # Validate the resnet_version
        valid_resnet_versions = models.__dict__.keys()
        if resnet_version not in valid_resnet_versions or not resnet_version.startswith('resnet'):
            raise ValueError(f"Invalid resnet_version: {resnet_version}. Must be one of "
                             f"the ResNet models available in torchvision.models (e.g., 'resnet18').")

        # Load the pre-trained or base ResNet model
        self.resnet = models.__dict__[resnet_version](pretrained=pretrained)

        # Replace the final fully connected layer to match the number of classes
        # Get the number of input features for the existing classifier layer
        num_ftrs = self.resnet.fc.out_features

        # Add classifier layer
        self.class_lin: nn.Linear = nn.Linear(num_ftrs, num_classes)

        # Note: The 'image_size' parameter is not explicitly used to modify the model's
        # structure here, as standard torchvision ResNets are fully convolutional
        # (except the final FC layer) and can handle variable input sizes up to a point.
        # It's included in the signature as per the request and could be used for
        # input validation or specific resizing logic if needed.

    #
    def get_embedding(self, x: Tensor) -> Tensor:

        #
        return self.resnet(x)


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
        x = self.resnet(x)
        #
        x = self.class_lin(x)
        #
        return x
