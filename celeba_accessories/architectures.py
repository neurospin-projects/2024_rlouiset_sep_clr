from torchvision.models import resnet18
from torch import nn

class Encoder(nn.Module):
    def __init__(self, common_dim, salient_dim):
        super(Encoder, self).__init__()

        self.common_dim = common_dim
        self.salient_dim = salient_dim

        # encoder
        self.common_enc = resnet18(pretrained=False)
        self.specific_enc = resnet18(pretrained=False)
        self.feature_dim = 512

        self.common_enc.fc = nn.Sequential(nn.Linear(self.feature_dim, common_dim))

        self.specific_enc.fc = nn.Sequential(nn.Linear(self.feature_dim, salient_dim))

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