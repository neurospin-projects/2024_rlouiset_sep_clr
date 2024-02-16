from torch.optim import Adam
import torch

from architectures import Encoder
from data_augmentation import DigitGridDataAugmentation
from dataset import get_dsprites_on_digit_grid_datasets
from losses import uniform_loss, align_loss, joint_entropy_loss, discrete_sup_align_loss, sup_align_loss

# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def norm(x):
    return torch.nn.functional.normalize(x, dim=1, p=2)

# instantiate hyper-parameters
n_epochs = 251
batch_size = 1024
common_size = 32
salient_size = 5
salient_reg_beta = 100
salient_term_beta = 100
jem_lambda = 10

def train(student, train_loader, epoch):
    student.train()
    train_loss = 0
    for batch_idx, (view, labels, bg_tg_label) in enumerate(train_loader):
        with torch.no_grad():
            view = view.cuda()
            labels = labels.cuda()
            bg_tg_label = bg_tg_label.cuda()

        # compute a distorted view for background image
        distorted_view = data_aug_pipeline(view)

        # compute common and salient spaces
        common_latents, specific_latents = student(view.cuda())
        distorted_common_latents, _ = student(distorted_view.cuda())

        # hard code the fact that f_{\theta_S}(x) = s'
        specific_latents_teacher = torch.clone(specific_latents)
        specific_latents_teacher[bg_tg_label==0] = 0.0

        # supervised attribute alignment for shape attribute
        sp_loss = uniform_loss(specific_latents_teacher[:, 0][:, None])
        sp_loss = sp_loss + (discrete_sup_align_loss(specific_latents[bg_tg_label==1, 0], labels[bg_tg_label==1, 0]) +
                       align_loss(specific_latents_teacher[bg_tg_label==0, 0][:, None], specific_latents[bg_tg_label==0, 0][:, None])) / 2
        # supervised attribute alignment for shape attribute
        for i in range(1,5):
            sp_loss = sp_loss + (sup_align_loss(specific_latents[bg_tg_label==1,i], labels[bg_tg_label==1, i]) +
                           align_loss(specific_latents_teacher[bg_tg_label==0, i][:, None], specific_latents[bg_tg_label==0, i][:, None], tau=0.025)) / 2
            sp_loss = sp_loss + uniform_loss(specific_latents_teacher[:,i][:, None])
        # normalize the salient loss
        sp_loss = sp_loss / 5.0

        # common term
        cm_loss = align_loss(norm(common_latents), norm(distorted_common_latents))
        cm_loss = cm_loss + (uniform_loss(norm(common_latents)) + uniform_loss(norm(distorted_common_latents))) / 2

        # null mutual information constraint
        jem_loss = joint_entropy_loss(norm(common_latents), specific_latents_teacher)

        # aggregate the losses
        loss = cm_loss + salient_term_beta * sp_loss + jem_lambda * jem_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss / len(train_loader)

# get datasets and data loaders
train_dataset, test_dataset = get_dsprites_on_digit_grid_datasets()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

# instantiate Data Augmentation pipeline
data_aug_pipeline = DigitGridDataAugmentation()

# instantiate the contrastive encoder
encoder = Encoder(common_dim=common_size, salient_dim=salient_size).float().cuda()

# instantiate the optimizer
optimizer = Adam(list(encoder.parameters()), lr=3e-4)

for epoch in range(1, n_epochs):
    print(train(encoder, train_loader, epoch))
torch.save(encoder, "dsprites-on-digit-grid_k-cem_dis_sep_clr.pth")
