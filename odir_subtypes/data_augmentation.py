from kornia.augmentation import ColorJitter, RandomGrayscale, RandomResizedCrop, RandomVerticalFlip, RandomRotation
from torchvision import transforms
from torch import Tensor
import torch.nn as nn
import torch

class ODIRDataAugmentation(nn.Module):
    """Module to perform data augmentation on eye fundus images using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        # color distrortion function
        s = 0.5
        jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        random_jitter = transforms.RandomApply([jitter], p=0.8)
        random_greyscale = RandomGrayscale(p=0.2)
        color_distort = nn.Sequential(random_jitter, random_greyscale)

        self.transforms = nn.Sequential(
            RandomResizedCrop((224, 224), scale=(0.75, 1)),
            RandomVerticalFlip(p=0.5),
            color_distort,
            RandomRotation(degrees=45),
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)
        return x_out