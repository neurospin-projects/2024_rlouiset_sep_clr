import numpy as np
import torch
from keras.datasets import cifar10
from torch.utils.data import Dataset
import tensorflow.compat.v2 as tf

class CifarMNISTDataset(Dataset):
    def __init__(self, data, cifar_targets, digit_targets, background_targets):
        self.data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2)).astype(float)).float()
        self.cifar_targets = torch.from_numpy(cifar_targets.astype(float)).float()
        self.digit_targets = torch.from_numpy(digit_targets.astype(float)).float()
        self.background_targets = torch.from_numpy(background_targets.astype(float)).float()

    def __len__(self):
        return len(self.cifar_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, cifar_labels, digit_label, background_target_label).
        """
        img, cifar_target, digit_target, background_target_label = self.data[index], int(self.cifar_targets[index]), int(self.digit_targets[index]), int(self.background_targets[index])

        return img, cifar_target, digit_target, background_target_label

def get_cifar_mnist_datasets():
    # import CIFAR-10
    (X_train_cifar, y_train_cif), (X_test_cifar, y_test_cif) = cifar10.load_data()
    X_train_cifar, y_train_cif = X_train_cifar[:50000] , y_train_cif[:50000]
    X_test_cifar, y_test_cif = X_test_cifar[:1000] , y_test_cif[:1000]

    # import MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train_mnist, y_train_mni), (X_test_mnist, y_test_mni) = mnist.load_data()
    X_train_mnist, y_train_mni = X_train_mnist[:50000], y_train_mni[:50000]
    X_train_mnist = np.array([np.pad(img, (2, 2)) for img in X_train_mnist])
    X_test_mnist, y_test_mni = X_test_mnist[:1000] , y_test_mni[:1000]
    X_test_mnist = np.array([np.pad(img, (2, 2)) for img in X_test_mnist])

    # build TRAIN dataset
    X_train = []
    y_train = []
    y_train_mnist = []
    y_train_cifar = []
    for i in range(len(X_train_mnist)):
        if i < 25000:
            X_train.append(0.5 * X_train_cifar[i] / 255.)
            y_train.append(0)
            y_train_mnist.append(-1)
            y_train_cifar.append(y_train_cif[i][0])
        else:
            X_train.append((0.5 * X_train_cifar[i] + 0.5 * np.repeat(X_train_mnist[i][:, :, None], 3, axis=2)) / 255.)
            y_train.append(1)
            y_train_mnist.append(y_train_mni[i])
            y_train_cifar.append(y_train_cif[i][0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train_mnist = np.array(y_train_mnist)
    y_train_cifar = np.array(y_train_cifar)

    # build TEST dataset
    X_test = []
    y_test = []
    y_test_mnist = []
    y_test_cifar = []
    for i in range(len(X_test_mnist)):
        if i < 500:
            X_test.append(0.5 * X_test_cifar[i] / 255.)
            y_test.append(0)
            y_test_mnist.append(-1)
            y_test_cifar.append(y_test_cif[i])
        else:
            X_test.append((0.5 * X_test_cifar[i] + 0.5 * np.repeat(X_test_mnist[i][:, :, None], 3, axis=2)) / 255.)
            y_test.append(1)
            y_test_mnist.append(y_test_mni[i])
            y_test_cifar.append(y_test_cif[i])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_test_mnist = np.array(y_test_mnist)
    y_test_cifar = np.array(y_test_cifar)

    # Instantiate MNIST superimposed on CIFAR-10 datasets
    train_dataset = CifarMNISTDataset(X_train, y_train_cifar, y_train_mnist, y_train)
    test_dataset = CifarMNISTDataset(X_test, y_test_cifar, y_test_mnist, y_test)

    return train_dataset, test_dataset