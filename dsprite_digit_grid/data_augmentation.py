from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGaussianNoise, RandomRotation
from torch import nn, Tensor
import torch

class DigitGridDataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        # color distrortion function
        self.transforms = nn.Sequential(
            RandomResizedCrop((64, 64), scale=(0.5, 1)),
            RandomRotation(degrees=25),
            RandomHorizontalFlip(p=0.5),
            RandomGaussianNoise()
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)
        return x_out