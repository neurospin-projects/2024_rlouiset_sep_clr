from sklearn.linear_model import LogisticRegression
import numpy as np
import torch

def evaluate_representations(encoder, train_loader, test_loader):
    common_size = encoder.common_dim
    encoder.eval()
    with torch.no_grad():
        X_train_mu = []
        X_test_mu = []
        y_train = []
        y_test = []
        y_digit_train = []
        y_digit_test = []
        y_cifar_train = []
        y_cifar_test = []
        for data, y_cifar, y_digit, y_binary in train_loader:
            data = data.cuda()
            common_rep, _, specific_rep, _ = encoder(data)
            mean = torch.cat((common_rep, specific_rep), dim=1)
            X_train_mu.extend(mean.cpu().numpy())
            y_train.extend(y_binary.cpu().numpy())
            y_digit_train.extend(y_digit.cpu().numpy())
            y_cifar_train.extend(y_cifar.cpu().numpy())
        for data, y_cifar, y_digit, y_binary in test_loader:
            data = data.cuda()
            common_rep, _, specific_rep, _ = encoder(data)
            mean = torch.cat((common_rep, specific_rep), dim=1)
            X_test_mu.extend(mean.cpu().numpy())
            y_test.extend(y_binary.cpu().numpy())
            y_digit_test.extend(y_digit.cpu().numpy())
            y_cifar_test.extend(y_cifar.cpu().numpy())
        X_train_mu = np.array(X_train_mu)
        X_test_mu = np.array(X_test_mu)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_cifar_train = np.array(y_cifar_train)
        y_cifar_test = np.array(y_cifar_test)
        y_digit_train = np.array(y_digit_train)
        y_digit_test = np.array(y_digit_test)

    # compute background classification
    log_reg = LogisticRegression().fit(X_train_mu[y_train == 1, common_size:], y_cifar_train[y_train == 1])
    log_reg_score = log_reg.score(X_test_mu[y_test == 1, common_size:], y_cifar_test[y_test == 1])
    print("specific trained on cifar labels : ", log_reg_score)

    # compute linear probe results
    log_reg = LogisticRegression().fit(X_train_mu[y_train == 1, common_size:], y_digit_train[y_train == 1])
    log_reg_score = log_reg.score(X_test_mu[y_test == 1, common_size:], y_digit_test[y_test == 1])
    print("specific trained on mnist labels : ", log_reg_score)

    # compute background classification
    log_reg = LogisticRegression().fit(X_train_mu[y_train == 1, :common_size], y_cifar_train[y_train == 1])
    log_reg_score = log_reg.score(X_test_mu[y_test == 1, :common_size], y_cifar_test[y_test == 1])
    print("common trained on cifar labels : ", log_reg_score)

    # compute linear probe results
    log_reg = LogisticRegression().fit(X_train_mu[y_train == 1, :common_size], y_digit_train[y_train == 1])
    log_reg_score = log_reg.score(X_test_mu[y_test == 1, :common_size], y_digit_test[y_test == 1])
    print("common trained on mnist labels : ", log_reg_score)