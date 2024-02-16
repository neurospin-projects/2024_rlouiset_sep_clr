import torch.nn.functional as F
from torch import nn

kernel_size = 4 # (4, 4) kernel filter
init_channels = 512 # initial number of filters
image_channels = 1 # number of image channels
hidden = 128 # number of hidden dimensions

class Encoder(nn.Module):
    def __init__(self, common_dim, salient_dim):
        super(Encoder, self).__init__()

        # common encoder convolutions
        self.common_enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.common_enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=256, kernel_size=kernel_size, stride=2, padding=1)
        self.common_enc3 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=kernel_size, stride=2, padding=1)
        self.common_enc4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=kernel_size, stride=2, padding=1)
        self.common_enc5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=kernel_size, stride=2, padding=1)

        # specific encoder convolutions
        self.specific_enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=256, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc3 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=kernel_size, stride=2, padding=1)

        # Add MLP projection from representation to latent space.
        self.common_projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, common_dim))
        self.specific_projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, salient_dim))

    def forward(self, x):
        batch, _, _, _ = x.shape

        # specific encoder
        h = F.relu(self.specific_enc1(x))
        h = F.relu(self.specific_enc2(h))
        h = F.relu(self.specific_enc3(h))
        h = F.relu(self.specific_enc4(h))
        h = F.relu(self.specific_enc5(h))
        h = h.reshape(batch, -1)
        specific_head = self.specific_projector(h)

        # common encoder
        h = F.relu(self.common_enc1(x))
        h = F.relu(self.common_enc2(h))
        h = F.relu(self.common_enc3(h))
        h = F.relu(self.common_enc4(h))
        h = F.relu(self.common_enc5(h))
        h = h.reshape(batch, -1)
        common_head = self.common_projector(h)

        return common_head, specific_head