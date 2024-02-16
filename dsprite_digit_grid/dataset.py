from torch.utils.data import Dataset
from random import shuffle
import tensorflow as tf
import numpy as np
import torch

class dSpritesDataset(Dataset):
    def __init__(self, data, salient_attributes, bg_tg_labels):
        self.data = torch.from_numpy(data.astype(float)).float()
        self.salient_attributes = torch.from_numpy(salient_attributes.astype(float)).float()
        self.bg_tg_labels = torch.from_numpy(bg_tg_labels.astype(float)).float()

    def __len__(self):
        return len(self.bg_tg_labels)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, salient_attributes, bg_tg_label = self.data[index], self.salient_attributes[index], self.bg_tg_labels[index]

        return img, salient_attributes, bg_tg_label

def get_dsprites_on_digit_grid_datasets():
    # load dsprite dataset
    dataset_zip = np.load('path/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']

    # shuffle and select the right rotation interval
    imgs, latents_values = shuffle(imgs, latents_values[:, [1,2,3,4,5]], random_state = 0)
    latents_values[:, 2][latents_values[:, 2] > (6.283185307179586/2)] = latents_values[:, 2][latents_values[:, 2] > (6.283185307179586/2)] - 6.283185307179586
    valid_rotation = (np.abs(latents_values[:, 2])<(6.283185307179586/8))
    imgs, latents_values = imgs[valid_rotation], latents_values[valid_rotation]

    # divide into train and test split for dsprite elements
    X_train_ds, y_train_ds = imgs[:50000][:, None, :, :], latents_values[:50000]
    X_test_ds, y_test_ds = imgs[50000:55000][:, None, :, :], latents_values[50000:55000]

    # load MNIST datasets
    mnist = tf.keras.datasets.mnist
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
    X_train_mnist = np.array([np.pad(img, (2, 2)) for img in X_train_mnist])
    X_test_mnist = np.array([np.pad(img, (2, 2)) for img in X_test_mnist])

    # build TRAIN dataset
    X_train = []
    y_train = []
    for i in range(len(X_train_ds)):
        if i < 25000:
            img = np.zeros_like(X_train_ds[i][0])
            img[:32, :32] = X_train_mnist[y_train_mnist==1][np.random.randint(0, len(X_train_mnist[y_train_mnist==1])-1)]
            img[32:, :32] = X_train_mnist[y_train_mnist==2][np.random.randint(0, len(X_train_mnist[y_train_mnist==2])-1)]
            img[:32, 32:] = X_train_mnist[y_train_mnist==3][np.random.randint(0, len(X_train_mnist[y_train_mnist==3])-1)]
            img[32:, 32:] = X_train_mnist[y_train_mnist==4][np.random.randint(0, len(X_train_mnist[y_train_mnist==4])-1)]
            X_train.append(img / 255.)
            y_train.append(0)
        else:
            img = np.zeros_like(X_train_ds[i][0])
            img[:32, :32] = X_train_mnist[y_train_mnist==1][np.random.randint(0, len(X_train_mnist[y_train_mnist==1])-1)]
            img[32:, :32] = X_train_mnist[y_train_mnist==2][np.random.randint(0, len(X_train_mnist[y_train_mnist==2])-1)]
            img[:32, 32:] = X_train_mnist[y_train_mnist==3][np.random.randint(0, len(X_train_mnist[y_train_mnist==3])-1)]
            img[32:, 32:] = X_train_mnist[y_train_mnist==4][np.random.randint(0, len(X_train_mnist[y_train_mnist==4])-1)]
            img = img + X_train_ds[i][0] * 255.
            img[img>255] = 255.0
            X_train.append(img / 255.)
            y_train.append(1)
    X_train = np.array(X_train)[:, None, :, :]
    y_train = np.array(y_train)

    # build TEST dataset
    X_test = []
    y_test = []
    for i in range(len(X_test_ds)):
        if i < 2500:
            img = np.zeros_like(X_test_ds[i][0])
            img[:32, :32] = X_test_mnist[y_test_mnist==1][np.random.randint(0, len(X_test_mnist[y_test_mnist==1])-1)]
            img[32:, :32] = X_test_mnist[y_test_mnist==2][np.random.randint(0, len(X_test_mnist[y_test_mnist==2])-1)]
            img[:32, 32:] = X_test_mnist[y_test_mnist==3][np.random.randint(0, len(X_test_mnist[y_test_mnist==3])-1)]
            img[32:, 32:] = X_test_mnist[y_test_mnist==4][np.random.randint(0, len(X_test_mnist[y_test_mnist==4])-1)]
            X_test.append(img / 255.)
            y_test.append(0)
        else:
            img = np.zeros_like(X_test_ds[i][0])
            img[:32, :32] = X_test_mnist[y_test_mnist==1][np.random.randint(0, len(X_test_mnist[y_test_mnist==1])-1)]
            img[32:, :32] = X_test_mnist[y_test_mnist==2][np.random.randint(0, len(X_test_mnist[y_test_mnist==2])-1)]
            img[:32, 32:] = X_test_mnist[y_test_mnist==3][np.random.randint(0, len(X_test_mnist[y_test_mnist==3])-1)]
            img[32:, 32:] = X_test_mnist[y_test_mnist==4][np.random.randint(0, len(X_test_mnist[y_test_mnist==4])-1)]
            img = img + X_test_ds[i][0] * 255.
            img[img>255] = 255.0
            X_test.append(img / 255.)
            y_test.append(1)
    X_test = np.array(X_test)[:, None, :, :]
    y_test = np.array(y_test)

    train_dataset = dSpritesDataset(X_train, y_train_ds, y_train)
    test_dataset = dSpritesDataset(X_test, y_test_ds, y_test)

    return train_dataset, test_dataset
