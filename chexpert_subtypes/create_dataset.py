import pandas as pd
import torchvision
import numpy as np
import cv2

input_dir = "path"

img_dims = 224

disease_names = ["Cardiomegaly", "Edema", "Pleural Effusion"]
uninteresting_disease_names = ["Pneumonia", "Enlarged Cardiomediastinum", "Pneumothorax", "Lung Lesion", "Pleural Other", "Fracture", "Consolidation", "Atelectasis"]

to_pil = torchvision.transforms.ToPILImage()

# train dataset construction
X_train = np.zeros((0, img_dims, img_dims))
y_train = {"diagnosis": [],
           "ap_pa": [],
           "age": [],
           "sex": [],
           "acquisition_device": []
           }

count = [0.0, 0.0, 0.0, 0.0]

df_train = pd.read_csv(input_dir + "CheXpert-v1.0-small/train.csv").to_dict('records')
for i in range(len(df_train)):
    row = df_train[i]
    to_continue = False
    if not row["No Finding"] == 1.0:
        for key in uninteresting_disease_names:
            value = row[key]
            if value == 1 or value == -1:
                to_continue = True
    if to_continue:
        continue

    if not row["AP/PA"] == "AP":
        continue

    label = [0, 0, 0, 0]
    if row["Frontal/Lateral"] == "Frontal":
        if row["No Finding"] == 1.0:
            label[0] = 1
        else:
            if row["Cardiomegaly"]==1:
                label[1] = 1
            elif row["Cardiomegaly"]==-1:
                continue

            if row["Edema"]==1:
                label[2] = 1
            elif row["Edema"]==-1:
                continue

            if row["Pleural Effusion"]==1:
                label[3] = 1
            elif row["Pleural Effusion"]==-1:
                continue

        if np.nansum(label) == 0 or np.nansum(label) > 1:
            continue
        if count[np.nanargmax(label)] > 10000 :
            continue
        if np.nanargmax(label) > 0 and count[np.nanargmax(label)] > 3000 :
            continue

        count[np.nanargmax(label)] += 1

        img = cv2.imread(input_dir + row["Path"], cv2.IMREAD_GRAYSCALE)[10:-10, 10:-10]
        img = cv2.resize(img, dsize=(img_dims, img_dims), interpolation=cv2.INTER_AREA)
        img = np.asarray(img)
        img = img / 255.
        X_train = np.concatenate((X_train, np.array(img)[None, :, :]), axis=0)

        y_train["diagnosis"].append(label)
        y_train["ap_pa"].append(row["AP/PA"])
        y_train["sex"].append(row["Sex"])
        y_train["age"].append(row["Age"])
        y_train["acquisition_device"].append(row["Support Devices"])

print("------------")

for i in range(0, 4) :
    print(np.nansum(np.array(y_train["diagnosis"])[:, i]))

np.save("path/X_train_224.npy", X_train)
pd.DataFrame(y_train).to_csv('path/y_train.csv')
