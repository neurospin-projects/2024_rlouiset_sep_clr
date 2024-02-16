from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import sklearn
import torch

class CheXpertDataset(Dataset):
    def __init__(self, data, age_targets, sex_targets, background_targets):
        self.data = torch.from_numpy(data.astype(float)).float()
        self.age_targets = torch.from_numpy(age_targets.astype(float)).float()
        self.sex_targets = torch.from_numpy(sex_targets.astype(float)).float()
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
        img, age_target, sex_target, background_target_label = self.data[index], int(self.age_targets[index]), int(self.sex_targets[index]), int(self.background_targets[index])

        return img, age_target, sex_target,  background_target_label

def get_chest_xrays_datasets():
    # import CheXpert subset
    X = np.load('path/X_train_224.npy')[:, None, :, :].repeat(3, axis=1)
    y = pd.read_csv('path/y_train_224.csv')

    y_age, y_sex, y_diagnosis = np.array(y["age"]), np.array(y["sex"]), np.array(y["diagnosis"])

    # translate diagnosis and sex to floats
    diagnosis_mapping = {'[1, 0, 0, 0]': 0,
                         '[0, 1, 0, 0]': 1,
                         '[0, 0, 1, 0]': 2,
                         '[0, 0, 0, 1]': 3}
    sex_mapping = {"Female": 0, "Male": 1, "Unknown": 1}

    # apply translator
    y_sex = np.array([sex_mapping[sex] for sex in y_sex])
    y_diagnosis = np.array([diagnosis_mapping[d] for d in y_diagnosis])

    # divide into TRAIN and TEST splits
    X_train, X_test, idx_train, idx_test = sklearn.model_selection.train_test_split(X, range(len(X)), test_size=0.1, random_state=0)
    y_age_test, y_sex_test, y_diagnosis_test = y_age[idx_test], y_sex[idx_test], y_diagnosis[idx_test]
    y_age, y_sex, y_diagnosis = y_age[idx_train], y_sex[idx_train], y_diagnosis[idx_train]

    # Dataset and Data Loader
    train_dataset = CheXpertDataset(X_train, y_age, y_sex, y_diagnosis)
    test_dataset = CheXpertDataset(X_test, y_age_test, y_sex_test, y_diagnosis_test)
    return train_dataset, test_dataset