from torch.utils.data import Dataset
import numpy as np
import torch

class CelebADataset(Dataset):
    def __init__(self, data, sex_targets, background_targets):
        self.data = torch.from_numpy(data.astype(float)).float()
        self.sex_targets = torch.from_numpy(sex_targets.astype(float)).float()
        self.accesories_targets = torch.from_numpy(background_targets.astype(float)).float()

    def __len__(self):
        return len(self.accesories_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, sex_label, background_target_label).
        """
        img, sex_target, accesorie_label = self.data[index], int(self.sex_targets[index]), int(self.accesories_targets[index])

        return img, sex_target, accesorie_label

def celeba_accesories_dataset():
    X_train = np.load('path/X_train_celeba_balanced_128.npy').transpose(0, 3, 1, 2) / 255.
    X_test = np.load('path/X_test_celeba_balanced_128.npy').transpose(0, 3, 1, 2) / 255.

    y_train = np.load('path/y_train_subtype_balanced_128.npy')
    y_test = np.load('path/y_test_subtype_balanced_128.npy')

    y_train_sex = np.load('path/y_train_sex_balanced_128.npy')
    y_test_sex = np.load('path/y_test_sex_balanced_128.npy')

    train_dataset = CelebADataset(X_train, y_train_sex, y_train)
    test_dataset = CelebADataset(X_test, y_test_sex, y_test)

    return train_dataset, test_dataset