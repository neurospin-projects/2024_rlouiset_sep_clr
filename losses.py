import torch

def sup_align_loss(x, y, tau=0.025):
    return (torch.pdist(y[:, None], p=2).pow(2).mul(-1/(2*tau)).exp() * torch.pdist(x[:, None], p=2).pow(2)).mean()

def discrete_sup_align_loss(x, y):
    return (torch.pdist(y[:, None], p=2).pow(2).mul(-1/1e-5).exp() * torch.pdist(x[:, None], p=2).pow(2)).mean()

def align_loss(x, y, alpha=2, tau=0.5):
    return (x - y).norm(p=2, dim=1).pow(alpha).mul(1/(2*tau)).mean()

def uniform_loss(x, tau=0.5):
    return torch.pdist(x, p=2).pow(2).mul(-1/(2*tau)).exp().mean().log()

def joint_entropy_loss(c, s, tau=0.5):
    s_similarity_matrix = torch.cdist(s, s, p=2.0).pow(2).mul(-1/(2*tau)).exp()
    c_similarity_matrix = torch.cdist(c, c, p=2.0).pow(2).mul(-1/(2*tau)).exp()
    jem_loss = (s_similarity_matrix * c_similarity_matrix).mean(-1).log().mean()
    return jem_loss