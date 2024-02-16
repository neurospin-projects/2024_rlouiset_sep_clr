import pandas as pd
from torch.utils.data import Dataset
import tensorflow.compat.v2 as tf
import numpy as np
import torch

class ODIRSubtypeDataset(Dataset):
    def __init__(self, data, age_targets, sex_targets, lr_targets, subtype_targets, background_targets):
        self.data = torch.from_numpy(data.astype(float)).float()
        self.age_targets = torch.from_numpy(age_targets.astype(float)).float()
        self.sex_targets = torch.from_numpy(sex_targets.astype(float)).float()
        self.lr_targets = torch.from_numpy(lr_targets.astype(float)).float()
        self.subtype_targets = torch.from_numpy(subtype_targets.astype(float)).float()
        self.background_targets = torch.from_numpy(background_targets.astype(float)).float()

    def __len__(self):
        return len(self.background_targets)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, age_target, sex_target, lr_target, subtype_target, background_target_label = (
            self.data[index], int(self.age_targets[index]), int(self.sex_targets[index]), int(self.lr_targets[index]),
            int(self.subtype_targets[index]), int(self.background_targets[index]))

        return img, age_target, sex_target, lr_target, subtype_target, background_target_label

def get_odir_subtypes_datasets():
    # load data
    X_train = np.load('path/X_train.npy').transpose(0, 3, 1, 2)
    y_train = pd.read_csv('path/y_train.csv')

    X_test = np.load('path/X_test.npy').transpose(0, 3, 1, 2)
    y_test = pd.read_csv('path/y_test.csv')

    # normalize pixel values
    X_test = X_test - np.min(X_train)
    X_train = X_train - np.min(X_train)
    X_test = X_test / (0.001 + np.max(X_train))
    X_train = X_train / (0.001 + np.max(X_train))

    # load age, sex, left/right eye, diagnosis (healhty or disease) and subtype attribute
    y_train_age, y_train_sex, y_train_lr, y_train_diagnosis, y_train_subtype = np.array(y_train["age"]), y_train[
        "sex"], np.array(y_train["left_right"]), np.array(y_train["diagnosis"]), np.array(y_train["subtype"])
    y_test_age, y_test_sex, y_test_lr, y_test_diagnosis, y_test_subtype = np.array(y_test["age"]), y_test[
        "sex"], np.array(y_test["left_right"]), np.array(y_test["diagnosis"]), np.array(y_test["subtype"])

    # few unknown age values are replaced by the mean of the age distribution
    y_train_age[y_train_age == 1] = np.mean(y_train_age)
    y_test_age[y_test_age == 1] = np.mean(y_test_age)

    # translate strings to int for sex attributes
    sex_mapping = {"Female": 0.0, "Male": 1.0}
    y_train_sex = np.array([sex_mapping[sex] for sex in y_train_sex])
    y_test_sex = np.array([sex_mapping[sex] for sex in y_test_sex])

    # Instantiate datasets
    train_dataset = ODIRSubtypeDataset(X_train, y_train_age, y_train_sex, y_train_lr, y_train_subtype, y_train_diagnosis)
    test_dataset = ODIRSubtypeDataset(X_test, y_test_age, y_test_sex, y_test_lr, y_test_subtype, y_test_diagnosis)

    return train_dataset, test_dataset