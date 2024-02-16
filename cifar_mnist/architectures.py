from torchvision.models import resnet18
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, common_dim, salient_dim):
        super(Encoder, self).__init__()

        self.common_dim = common_dim
        self.salient_dim = salient_dim

        # encoder
        self.common_enc = resnet18(pretrained=False)
        self.specific_enc = resnet18(pretrained=False)
        self.feature_dim = 512

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.common_enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.common_enc.maxpool = nn.Identity()
        self.common_enc.fc = nn.Linear(self.feature_dim, common_dim)

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.specific_enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.specific_enc.maxpool = nn.Identity()
        self.specific_enc.fc = nn.Linear(self.feature_dim, salient_dim)

        # Add MLP projection.
        self.common_projector = nn.Sequential(nn.Linear(common_dim, 128),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Linear(128, common_dim))

        self.specific_projector = nn.Sequential(nn.Linear(salient_dim, 128),
                                              nn.BatchNorm1d(128),
                                              nn.ReLU(),
                                              nn.Linear(128, salient_dim))

    def forward(self, x):
        common_rep = self.common_enc(x)
        specific_rep = self.specific_enc(x)

        common_head = self.common_projector(common_rep)
        specific_head = self.specific_projector(specific_rep)

        return common_rep, common_head, specific_rep, specific_head