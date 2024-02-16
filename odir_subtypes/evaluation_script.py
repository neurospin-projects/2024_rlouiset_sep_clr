from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import torch

def evaluate_representations(encoder, train_loader, test_loader):
    common_size = encoder.common_dim
    encoder.eval()
    with torch.no_grad():
        X_train_rep = []
        X_test_rep = []
        X_train_head = []
        X_test_head = []
        y_train_age = []
        y_test_age = []
        y_train_sex = []
        y_test_sex = []
        y_train_lr = []
        y_test_lr = []
        y_train_subtype = []
        y_test_subtype = []
        y_train_diag = []
        y_test_diag = []
        for data, y_age, y_sex, y_lr, y_subtype, y in train_loader:
            data = data.cuda()
            common_rep, common_head, specific_rep, specific_head = encoder(data)
            mean = torch.cat((common_rep, specific_rep), dim=1)
            X_train_rep.extend(mean.cpu().numpy())
            head = torch.cat((common_head, specific_head), dim=1)
            X_train_head.extend(head.cpu().numpy())
            y_train_age.extend(y_age.cpu().numpy())
            y_train_sex.extend((y_sex>0.5).float().cpu().numpy())
            y_train_lr.extend(y_lr.cpu().numpy())
            y_train_subtype.extend(y_subtype)
            y_train_diag.extend(y.cpu().numpy())
        for data, y_age, y_sex, y_lr, y_subtype, y in test_loader:
            data = data.cuda()
            common_rep, common_head, specific_rep, specific_head = encoder(data)
            mean = torch.cat((common_rep, specific_rep), dim=1)
            X_test_rep.extend(mean.cpu().numpy())
            head = torch.cat((common_head, specific_head), dim=1)
            X_test_head.extend(head.cpu().numpy())
            y_test_age.extend(y_age.cpu().numpy())
            y_test_sex.extend((y_sex>0.5).float().cpu().numpy())
            y_test_lr.extend(y_lr.cpu().numpy())
            y_test_subtype.extend(y_subtype)
            y_test_diag.extend(y.cpu().numpy())
        X_train_rep = np.array(X_train_rep)
        X_test_rep = np.array(X_test_rep)
        X_train_head = np.array(X_train_head)
        X_test_head = np.array(X_test_head)
        y_train_age = np.array(y_train_age)
        y_test_age = np.array(y_test_age)
        y_train_lr = np.array(y_train_lr)
        y_test_lr = np.array(y_test_lr)
        y_train_sex = np.array(y_train_sex)
        y_test_sex = np.array(y_test_sex)
        y_train_subtype = np.array(y_train_subtype)
        y_test_subtype = np.array(y_test_subtype)
        y_train_diag = np.array(y_train_diag)
        y_test_diag = np.array(y_test_diag)

    # compute bg / tg perf
    log_reg = LogisticRegression().fit(X_train_head[:, :common_size], y_train_diag)
    log_reg_score = balanced_accuracy_score(y_test_diag, log_reg.predict(X_test_head[:, :common_size]))
    print("Linear probe trained on binary labels, common latents : ", log_reg_score)
    log_reg = LogisticRegression().fit(X_train_head[:, common_size:], y_train_diag)
    log_reg_score = balanced_accuracy_score(y_test_diag, log_reg.predict(X_test_head[:, common_size:]))
    print("Linear probe trained on binary labels, specific latents : ", log_reg_score)
    print('')

    # compute subtype perf
    log_reg = LogisticRegression().fit(X_train_rep[y_train_diag>0.5, common_size:], y_train_subtype[y_train_diag>0.5])
    log_reg_score = balanced_accuracy_score(y_test_subtype[y_test_diag>0.5], log_reg.predict(X_test_rep[y_test_diag>0.5, common_size:]))
    print("Linear probe trained on subtype labels, specific latents : ", log_reg_score)
    log_reg = LogisticRegression().fit(X_train_rep[y_train_diag>0.5, :common_size], y_train_subtype[y_train_diag>0.5])
    log_reg_score = balanced_accuracy_score(y_test_subtype[y_test_diag>0.5], log_reg.predict(X_test_rep[y_test_diag>0.5, :common_size]))
    print("Linear probe trained on subtype labels, common latents : ", log_reg_score)
    print('')

    # compute age perf
    log_reg = LinearRegression().fit(X_train_rep[y_train_diag>0.5, common_size:], y_train_age[y_train_diag>0.5])
    log_reg_score = np.mean(np.abs(y_test_age[y_test_diag>0.5] - log_reg.predict(X_test_rep[y_test_diag>0.5, common_size:])))
    print("Linear probe trained on age labels, specific latents : ", log_reg_score)
    log_reg = LinearRegression().fit(X_train_rep[y_train_diag>0.5, :common_size], y_train_age[y_train_diag>0.5])
    log_reg_score =  np.mean(np.abs(y_test_age[y_test_diag>0.5] - log_reg.predict(X_test_rep[y_test_diag>0.5, :common_size])))
    print("Linear probe trained on age labels, common latents : ", log_reg_score)
    print('')

    # compute sex perf
    log_reg = LogisticRegression().fit(X_train_rep[y_train_diag>0.5, common_size:], y_train_sex[y_train_diag>0.5])
    log_reg_score = balanced_accuracy_score(y_test_sex[y_test_diag>0.5], log_reg.predict(X_test_rep[y_test_diag>0.5, common_size:]))
    print("Linear probe trained on sex labels, specific latents : ", log_reg_score)
    log_reg = LogisticRegression().fit(X_train_rep[y_train_diag>0.5, :common_size], y_train_sex[y_train_diag>0.5])
    log_reg_score =  balanced_accuracy_score(y_test_sex[y_test_diag>0.5], log_reg.predict(X_test_rep[y_test_diag>0.5, :common_size]))
    print("Linear probe trained on sex labels, common latents : ", log_reg_score)
    print('')

    # compute lr perf
    log_reg = LogisticRegression().fit(X_train_rep[y_train_diag > 0.5, common_size:], y_train_lr[y_train_diag > 0.5])
    log_reg_score = balanced_accuracy_score(y_test_lr[y_test_diag > 0.5],  log_reg.predict(X_test_rep[y_test_diag > 0.5, common_size:]))
    print("Linear probe trained on lr labels, specific latents : ", log_reg_score)
    log_reg = LogisticRegression().fit(X_train_rep[y_train_diag > 0.5, :common_size], y_train_lr[y_train_diag > 0.5])
    log_reg_score = balanced_accuracy_score(y_test_lr[y_test_diag > 0.5],
                                            log_reg.predict(X_test_rep[y_test_diag > 0.5, :common_size]))
    print("Linear probe trained on lr labels, common latents : ", log_reg_score)
    print('')