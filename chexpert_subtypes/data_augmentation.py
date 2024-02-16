from kornia.augmentation import ColorJitter, RandomGrayscale, RandomResizedCrop, RandomHorizontalFlip, RandomRotation
from torchvision import transforms
from torch import Tensor
import torch.nn as nn
import torch

class ChestXRayDataAugmentation(nn.Module):
    """Module to perform data augmentation on Chest XRays images using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        # color distrortion function
        s = 0.5
        jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        random_jitter = transforms.RandomApply([jitter], p=0.8)
        random_greyscale = RandomGrayscale(p=0.2)
        color_distort = nn.Sequential(random_jitter, random_greyscale)

        self.transforms = nn.Sequential(
            RandomResizedCrop((224, 224), scale=(0.2, 1)),
            color_distort,
            RandomRotation(degrees=45),
            RandomHorizontalFlip(p=0.5),
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)
        return x_out