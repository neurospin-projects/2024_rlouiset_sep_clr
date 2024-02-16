from torch.optim import Adam
import torch

from architectures import Encoder
from data_augmentation import CifarMNISTDataAugmentation
from dataset import get_cifar_mnist_datasets
from evaluation_script import evaluate_representations
from losses import align_loss, uniform_loss, joint_entropy_loss

# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import random
random.seed(0)

# instantiate the hyper-parameters
batch_size = 512
n_epochs = 251
common_size=32
salient_size=32
salient_reg_beta = 1000
salient_term_beta = 1000
jem_lambda = 10

def norm(x):
    return torch.nn.functional.normalize(x, dim=1, p=2)

def train(encoder, train_loader):
    encoder.train()
    train_loss = 0
    for batch_idx, (view, _, _, labels) in enumerate(train_loader):
        with torch.no_grad():
            view1 = data_aug_pipeline(view)
            view2 = data_aug_pipeline(view)
            labels = labels.cuda()

        # compute common and salient spaces
        _, common_head_1, _, specific_head_1 = encoder(view1.cuda())
        _, common_head_2, _, specific_head_2 = encoder(view2.cuda())

        # hard code the fact that f_{\theta_S}(x) = s'
        specific_head_1_teacher = torch.clone(specific_head_1)
        specific_head_2_teacher = torch.clone(specific_head_2)
        specific_head_1_teacher[labels==0, :] = 0.0 # s' = 0.0, i.e: the null vector
        specific_head_2_teacher[labels==0, :] = 0.0 # s' = 0.0, i.e: the null vector

        # common term
        cm_align_loss = align_loss(norm(common_head_1), norm(common_head_2))
        cm_uniform_loss = (uniform_loss(norm(common_head_2)) + uniform_loss(norm(common_head_1))) / 2.0
        cm_loss = cm_align_loss + 2.0 * cm_uniform_loss

        # salient term
        sp_align_loss = align_loss(specific_head_1[labels==1], specific_head_2[labels==1])
        sp_uniform_loss = (uniform_loss(specific_head_2_teacher) + uniform_loss(specific_head_1_teacher)) / 2.0
        sp_loss = sp_align_loss + 2.0 * sp_uniform_loss

        # information-less constraint
        sp_reg_loss = (align_loss(specific_head_1_teacher[labels==0], specific_head_1[labels==0]) +
                         align_loss(specific_head_2_teacher[labels==0], specific_head_2[labels==0])) / 2.0

        # null mutual information constraint
        jem_loss = joint_entropy_loss(norm(common_head_1), specific_head_1_teacher)
        jem_loss = jem_loss + joint_entropy_loss(norm(common_head_2), specific_head_2_teacher)
        jem_loss = jem_loss / 2.0

        # aggregate the losses
        loss = cm_loss + salient_term_beta*sp_loss + salient_reg_beta*sp_reg_loss + jem_lambda*jem_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

# get datasets and data loaders
train_dataset, test_dataset = get_cifar_mnist_datasets()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

# instantiate Data Augmentation pipeline
data_aug_pipeline = CifarMNISTDataAugmentation()

# instantiate the contrastive encoder
encoder = Encoder(common_dim=common_size, salient_dim=salient_size).float().cuda()

# instantiate the optimizer
optimizer = Adam(list(encoder.parameters()), lr=3e-4)

# train and evaluate the method
for epoch in range(1, n_epochs):
    train(encoder, train_loader)
    if epoch % 50 == 0:
        evaluate_representations(encoder, train_loader, test_loader)
        print("---")
evaluate_representations(encoder, train_loader, test_loader)

torch.save(encoder, "cifar-mnist_k-jem_sep_clr.pth")
