import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

input_dir = "path/odir_5k/"

max_count_per_labels = [2300, 400, 350, 350, 350, 350]
count_per_labels = [0, 0, 0, 0, 0, 0]
target_to_label = {"N": 0, "D": 1, "G": 2, "C": 3, "A": 4, "M": 5}
img_dims = 224

# list_of_training_metadata = pd.read_csv(input_dir + "data.xlsx").to_dict('records')
list_of_metadata = pd.read_excel(input_dir + "data.xlsx", engine='openpyxl').to_dict('records')

# train dataset construction
X_ = np.zeros((0, img_dims, img_dims, 3))
y_ = {"diagnosis": [], "subtype": [], "age": [], "sex": [], "left_right": []}

for i, metadata in enumerate(list_of_metadata):
    for lr, which_eye in enumerate(['Left-Fundus', 'Right-Fundus']):
        if os.path.isfile(input_dir + "preprocessed_images/" + metadata[which_eye]):
            img = cv2.imread(input_dir + "preprocessed_images/" + metadata[which_eye])
        else:
            continue

        target = [0] * 6
        for j, key in enumerate(target_to_label.keys()) :
            target[j] = metadata[key]
        target = np.array(target)
        if np.sum(target) == 1:
            target = np.argmax(target)
        else:
            continue

        if count_per_labels[target] < max_count_per_labels[target] :
            count_per_labels[target] += 1
        else :
            continue

        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        img = (img - __mean__) / __std__
        X_ = np.concatenate((X_, np.array(img)[None]), axis=0)

        y_["diagnosis"].append(int(target>0.5))
        y_["subtype"].append(target-1)
        y_["age"].append(metadata["Patient Age"])
        y_["sex"].append(metadata["Patient Sex"])
        y_["left_right"].append(lr)

X_val = X_[:3]
y_val = {key: y_[key][:3] for key in y_.keys()}

X__ = X_[3:]
y__ = {key: np.array(y_[key][3:]) for key in y_.keys()}

X_train, X_test, idx_train, idx_test = train_test_split(X__, range(0, len(X__)), test_size=0.1, stratify=y__["diagnosis"])
y_train = {key: y__[key][idx_train] for key in y__.keys()}
y_test = {key: y__[key][idx_test] for key in y__.keys()}

print(X_test.shape)
print(X_train.shape)

y_train_numpy = np.array(y_train["subtype"])
y_test_numpy = np.array(y_test["subtype"])

print(y_test_numpy)

print('-------------')

for i in range(-1, 5) :
    print(np.sum(y_train_numpy==i))
for i in range(-1, 5):
    print(np.sum(y_test_numpy == i))

np.save("path/odir_5k/X_val.npy", X_val)
pd.DataFrame(y_val).to_csv('path/odir_5k/y_val.csv')

np.save("path/odir_5k/X_test.npy", X_test)
pd.DataFrame(y_test).to_csv('path/odir_5k/y_test.csv')

np.save("path/odir_5k/X_train.npy", X_train)
pd.DataFrame(y_train).to_csv('path/odir_5k/y_train.csv')
