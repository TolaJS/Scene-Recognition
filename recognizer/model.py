from torchvision import models
from safetensors.torch import load_model, save_model
from pthflops import count_ops
import torch.nn as nn
import torch


class ResNetFineTunedClassifier(nn.Module):
    def __init__(self, num_classes: int = 40, freeze_pretrained_weights: bool = True):
        """
        Initializes the ResNetFineTunedClassifier.

        Parameters:
            num_classes (int): Number of classes for classification. Default is 40.
            freeze_pretrained_weights (bool): Whether to freeze the pretrained weights of the ResNet model. Default is True.
        """
        super(ResNetFineTunedClassifier, self).__init__()
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        if freeze_pretrained_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def trainable_parameters(self):
        """
        Return the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """
        Save the model to the specified path.

        Parameters:
            path (str): The path where the model will be saved.

        Returns:
            None
        """
        save_model(self, path)

    def load(self, path: str) -> None:
        """
        Load the model from the specified path.

        Parameters:
            path (str): The path to load the model from.

        Returns:
            None
        """
        load_model(self, path)

    def no_of_params(self) -> int:
        """
        A function that calculates the total number of parameters in a model.
        Returns:
            int: The total number of parameters.
        """
        _no_params = 0
        for p in self.model.parameters():
            _no_params += p.nelement()
        return _no_params

    def size_in_memory(self) -> int:
        """
        Calculate the total size in memory taken by the model parameters and buffers.
        This function does not take any parameters and returns an integer representing the total size in bytes.
        """
        _param_size = 0
        for p in self.model.parameters():
            _param_size += p.nelement() * p.element_size()
        buffer_size = sum([buf.nelement() * buf.element_size() for buf in self.model.buffers()])
        return _param_size + buffer_size

    def no_of_flops(self, dtype: torch.dtype = torch.float32) -> int:
        """
        Calculate the number of floating-point operations for the given model and input data.

        Parameters:
            dtype (torch.dtype): The data type for the input data. Default is torch.float32.

        Returns:
            int: The total number of floating-point operations.
        """
        return count_ops(self.model, torch.randn(1, 3, 224, 224).to(dtype=dtype))
