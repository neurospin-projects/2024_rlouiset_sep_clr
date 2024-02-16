import math
import numpy as np
import pandas as pd
from glob import glob
import random
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import shutil

import sklearn.model_selection
from PIL import Image
from tqdm import tqdm

def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download_extract(database_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_CELEBA_NAME = 'celeba'

    if database_name == DATASET_CELEBA_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        extract_path = os.path.join(data_path, 'img_align_celeba')
        save_path = os.path.join(data_path, 'celeba.zip')
        extract_fn = _unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)

def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 176
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch


def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
        images[:save_size * save_size],
        (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im

def filter_images_by_attribute(data_dir, attr1=None, attr2=None, present1=True, present2=True):
    if attr1 is None and attr2 is None:
        return glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))
    df = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))
    assert attr1 in df.columns
    assert attr2 in df.columns
    val1 = 1 if present1 else -1
    val2 = 1 if present2 else -1
    df = df.loc[(df[attr1] == val1) & (df[attr2] == val2)]
    image_ids = df['File_name'].values
    image_ids = [i for i in image_ids]
    return image_ids

data_dir = '/home/robin/Desktop/rl264746/celeba_data/'

random.seed(0)

attr = 'Eyeglasses'
reverse = False
ratio = 0

width = 128
height = 128

# Images with only glasses
glasses_ids = filter_images_by_attribute(
    data_dir=data_dir,
    attr1='Eyeglasses',
    present1=True,
    attr2='Wearing_Hat',
    present2=False
)

hat_ids = filter_images_by_attribute(
    data_dir=data_dir,
    attr1='Eyeglasses',
    present1=False,
    attr2='Wearing_Hat',
    present2=True
)

bg_ids = filter_images_by_attribute(
    data_dir=data_dir,
    attr1='Eyeglasses',
    present1=False,
    attr2='Wearing_Hat',
    present2=False
)

df = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))
path = 'path/celeba_data/img_align_celeba/'

ids = df["File_name"].tolist()
sex = df["Male"].tolist()


"""X = np.zeros((0, 128, 128, 3))
y_subtype = []
y_sex = []
# glasses dataset
for i, id in enumerate(glasses_ids) :
    if np.sum(np.array(y_sex)[np.array(y_subtype)==1]==sex[ids.index(id)]) < 3000:
        img = get_image(path+id, width, height, 'RGB')
        X = np.concatenate((X, np.array(img)[None,:,:,:]))
        y_subtype.append(1)
        y_sex.append(sex[ids.index(id)])
    if np.sum((np.array(y_subtype) == 1)) % 100 == 0:
        print(np.sum((np.array(y_subtype) == 1)))
    if np.sum((np.array(y_subtype) == 1)) == 6000:
        break
np.save("/home/robin/Desktop/rl264746/celeba_data/X_gls_celeba_balanced_128.npy", X)
np.save("/home/robin/Desktop/rl264746/celeba_data/y_gls_subtype_balanced_128.npy", np.array(y_subtype))
np.save("/home/robin/Desktop/rl264746/celeba_data/y_gls_sex_balanced_128.npy", np.array(y_sex))"""

"""X = np.zeros((0, 128, 128, 3))
y_subtype = []
y_sex = []
# hats dataset
for i, id in enumerate(hat_ids) :
    if np.sum(np.array(y_sex)[np.array(y_subtype)==2]==sex[ids.index(id)]) < 3000:
        img = get_image(path + id, width, height, 'RGB')
        X = np.concatenate((X, np.array(img)[None,:,:,:]))
        y_subtype.append(2)
        y_sex.append(sex[ids.index(id)])
    if np.sum((np.array(y_subtype) == 2)) % 100 == 0:
        print(np.sum((np.array(y_subtype) == 2)))
    if np.sum((np.array(y_subtype) == 2)) == 6000:
        break
np.save("path/celeba_data/X_hats_celeba_balanced_128.npy", X)
np.save("path/rl264746/celeba_data/y_hats_subtype_balanced_128.npy", np.array(y_subtype))
np.save("path/rl264746/celeba_data/y_hats_sex_balanced_128.npy", np.array(y_sex))

X = np.zeros((0, 128, 128, 3))
y_subtype = []
y_sex = []

# background dataset
for i, id in enumerate(bg_ids):
    if np.sum(np.array(y_sex)[np.array(y_subtype) == 0] == sex[ids.index(id)]) < 6000:
        img = get_image(path + id, width, height, 'RGB')
        X = np.concatenate((X, np.array(img)[None, :, :, :]))
        y_subtype.append(0)
        y_sex.append(sex[ids.index(id)])
    if np.sum((np.array(y_subtype) == 0)) % 100 == 0:
        print(np.sum((np.array(y_subtype) == 0)))
    if np.sum((np.array(y_subtype) == 0)) == 12000:
        break

np.save("path/celeba_data/X_bg_celeba_balanced_128.npy", X)
np.save("path/celeba_data/y_bg_subtype_balanced_128.npy", np.array(y_subtype))
np.save("path/celeba_data/y_bg_sex_balanced_128.npy", np.array(y_sex))"""


X_gls = np.load("path/celeba_data/X_gls_celeba_balanced_128.npy")
y_gls_subtype = np.load("path/celeba_data/y_gls_subtype_balanced_128.npy")
y_sex_gls = np.load("path/celeba_data/y_gls_sex_balanced_128.npy")

X_hats = np.load("path/celeba_data/X_hats_celeba_balanced_128.npy")
y_hats_subtype = np.load("path/celeba_data/y_hats_subtype_balanced_128.npy")
y_sex_hats = np.load("path/celeba_data/y_hats_sex_balanced_128.npy")

X = np.load("path/celeba_data/X_bg_celeba_balanced_128.npy")
y_subtype = np.load("path/celeba_data/y_bg_subtype_balanced_128.npy")
y_sex = np.load("path/celeba_data/y_bg_sex_balanced_128.npy")

y_sex = np.concatenate((y_sex, y_sex_gls, y_sex_hats), axis=0)
y_subtype = np.concatenate((y_subtype, y_gls_subtype, y_hats_subtype), axis=0)
X = np.concatenate((X, X_gls, X_hats), axis=0)

print("y=0")
print(np.sum(np.array(y_sex)[np.array(y_subtype)==0]==-1))
print(np.sum(np.array(y_sex)[np.array(y_subtype)==0]==1))

print("y=1")
print(np.sum(np.array(y_sex)[np.array(y_subtype)==1]==-1))
print(np.sum(np.array(y_sex)[np.array(y_subtype)==1]==1))

print("y=2")
print(np.sum(np.array(y_sex)[np.array(y_subtype)==2]==-1))
print(np.sum(np.array(y_sex)[np.array(y_subtype)==2]==1))

X_train, X_test, idx_train, idx_test = sklearn.model_selection.train_test_split(X, range(len(X)), test_size=0.2)

np.save("path/celeba_data/X_train_celeba_balanced_128.npy", X_train)
np.save("path/celeba_data/X_test_celeba_balanced_128.npy", X_test)

np.save("path/y_train_subtype_balanced_128.npy", np.array(y_subtype)[idx_train])
np.save("path/y_test_subtype_balanced_128.npy", np.array(y_subtype)[idx_test])

np.save("path/y_train_sex_balanced_128.npy", np.array(y_sex)[idx_train])
np.save("path/y_test_sex_balanced_128.npy", np.array(y_sex)[idx_test])
