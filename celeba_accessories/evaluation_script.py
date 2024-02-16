from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

def evaluate_representations(encoder, train_loader, test_loader):
    common_size = encoder.common_dim
    with torch.no_grad():
        X_train_mu = []
        X_test_mu = []
        y_train_sex = []
        y_test_sex = []
        y_train_subtype = []
        y_test_subtype = []
        for data, y_sex, y in train_loader:
            data = data.cuda()
            common_rep, _, specific_rep, _ = encoder(data)
            mean = torch.cat((common_rep, specific_rep), dim=1)
            X_train_mu.extend(mean.cpu().numpy())
            y_train_sex.extend((y_sex>0.5).float().cpu().numpy())
            y_train_subtype.extend(y.cpu().numpy())
        for data, y_sex, y in test_loader:
            data = data.cuda()
            common_rep, _, specific_rep, _ = encoder(data)
            mean = torch.cat((common_rep, specific_rep), dim=1)
            X_test_mu.extend(mean.cpu().numpy())
            y_test_sex.extend((y_sex>0.5).float().cpu().numpy())
            y_test_subtype.extend(y.cpu().numpy())
        X_train_mu = np.array(X_train_mu)
        X_test_mu = np.array(X_test_mu)
        y_train_sex = np.array(y_train_sex)
        y_test_sex = np.array(y_test_sex)
        y_train_subtype = np.array(y_train_subtype)
        y_test_subtype = np.array(y_test_subtype)

    # compute performance to discriminate people with and without accesories
    log_reg = LogisticRegression().fit(X_train_mu[:, :common_size], (y_train_subtype>0.5).astype(float))
    log_reg_score = balanced_accuracy_score((y_test_subtype>0.5).astype(float), log_reg.predict(X_test_mu[:, :common_size]))
    print("Linear probe trained on binary labels, common latents : ", log_reg_score)

    log_reg = LogisticRegression().fit(X_train_mu[:, common_size:], (y_train_subtype>0.5).astype(float))
    log_reg_score = balanced_accuracy_score((y_test_subtype>0.5).astype(float), log_reg.predict(X_test_mu[:, common_size:]))
    print("Linear probe trained on binary labels, specific latents : ", log_reg_score)

    # compute performance to discriminate the target subtype: hats or glasses accesorie
    log_reg = LogisticRegression().fit(X_train_mu[y_train_subtype>0.5, common_size:], y_train_subtype[y_train_subtype>0.5])
    log_reg_score = balanced_accuracy_score(y_test_subtype[y_test_subtype>0.5], log_reg.predict(X_test_mu[y_test_subtype>0.5, common_size:]))
    print("Linear probe trained on subtype labels, specific latents : ", log_reg_score)

    log_reg = LogisticRegression().fit(X_train_mu[y_train_subtype>0.5, :common_size], y_train_subtype[y_train_subtype>0.5])
    log_reg_score = balanced_accuracy_score(y_test_subtype[y_test_subtype>0.5], log_reg.predict(X_test_mu[y_test_subtype>0.5, :common_size]))
    print("Linear probe trained on subtype labels, common latents : ", log_reg_score)

    # compute performance to discriminate male or female
    log_reg = LogisticRegression().fit(X_train_mu[y_train_subtype>0.5, common_size:], y_train_sex[y_train_subtype>0.5])
    log_reg_score = balanced_accuracy_score(y_test_sex[y_test_subtype>0.5], log_reg.predict(X_test_mu[y_test_subtype>0.5, common_size:]))
    print("Linear probe trained on sex labels, specific latents : ", log_reg_score)
    log_reg = LogisticRegression().fit(X_train_mu[y_train_subtype>0.5, :common_size], y_train_sex[y_train_subtype>0.5])
    log_reg_score =  balanced_accuracy_score(y_test_sex[y_test_subtype>0.5], log_reg.predict(X_test_mu[y_test_subtype>0.5, :common_size]))
    print("Linear probe trained on sex labels, common latents : ", log_reg_score)