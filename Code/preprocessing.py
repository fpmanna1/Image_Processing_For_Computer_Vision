import os
from sys import platform

import time

import cv2

import numpy as np

import imageio
import skimage.io as io
import matplotlib.pyplot as plt

import csv

import rawpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, functional

from PIL import Image

from glob import glob
from tqdm import tqdm

# Default directory separator used in dataset's text files
DEFAULT_DIR_SEPARATOR = '/'

if platform == 'win32':
    ROOT_DIR = "C:\\Dataset_Sony"
elif platform in ['linux', 'linux2', 'darwin']:
    ROOT_DIR = "/Dataset_Sony"
else:
    ROOT_DIR = None
    raise OSError("Unsupported operating system")

DATA_DIR = os.path.join(ROOT_DIR, 'Sony')
TRAIN_PATHS = os.path.join(ROOT_DIR, 'Sony_train_list.txt')
VALID_PATHS = os.path.join(ROOT_DIR, 'Sony_val_list.txt')
TEST_PATHS = os.path.join(ROOT_DIR, 'Sony_test_list.txt')


def modifica_path(input_path, short, index):
    data_path = 'C:\\Dataset_Sony\\Sony\\'
    # Divisione del percorso in componenti
    directory, filename = os.path.split(input_path)
    # Sostituzione di "short" con "short_np" e ".ARW" con ".npy"
    nuovo_filename = filename.replace(".ARW", "_" + str(index) + ".npy")

    if short is False:
        directory = data_path + "numpy_long"
    else:
        directory = data_path + "numpy_short"

    # Creazione del nuovo percorso
    nuovo_path = os.path.join(directory, nuovo_filename)

    return nuovo_path


def segmenta_s(immagine, dimensione_patch, input_path):
    altezza, larghezza = immagine.shape[-2:]
    print(f'h{altezza} w{larghezza}')

    righe_patch = altezza // dimensione_patch
    colonne_patch = larghezza // dimensione_patch

    index = 0
    for r in range(righe_patch):
        inizio_riga = r * dimensione_patch
        fine_riga = inizio_riga + dimensione_patch

        for c in range(colonne_patch):
            inizio_colonna = c * dimensione_patch
            fine_colonna = inizio_colonna + dimensione_patch
            # prendi tutti i canali
            patch_corrente = immagine[:, inizio_riga:fine_riga, inizio_colonna:fine_colonna]

            # if patch_corrente.shape[:2] == (dimensione_patch, dimensione_patch):
            img_path = modifica_path(input_path, short=True, index=index)
            index += 1
            np.save(img_path, patch_corrente)


def segmenta_l(immagine, dimensione_patch, input_path):
    altezza, larghezza = immagine.shape[:2]

    righe_patch = altezza // dimensione_patch
    colonne_patch = larghezza // dimensione_patch

    index = 0
    for r in range(righe_patch):
        inizio_riga = r * dimensione_patch
        fine_riga = inizio_riga + dimensione_patch

        for c in range(colonne_patch):
            inizio_colonna = c * dimensione_patch
            fine_colonna = inizio_colonna + dimensione_patch

            patch_corrente = immagine[inizio_riga:fine_riga, inizio_colonna:fine_colonna]

            # if patch_corrente.shape[:2] == (dimensione_patch, dimensione_patch):
            img_path = modifica_path(input_path, short=False, index=index)
            index += 1
            np.save(img_path, patch_corrente)


class SidDataset(Dataset):
    def _init_(self, file_path, token, train_mode=False, auto_save=False, transform=None):
        # contains the paths of the short exposed, i.e. ground truth, and long exposed images, i.e. labels
        self.paths = []

        # contains the path of the file, which were all the paths are stored; short and long paths are on the same
        # row, each one on its individual column
        self.file_path = file_path

        # character delimiter, in order to understand of columns are separated in the file
        self.token = token

        # behaves differently when in train mode, since because all images must be patched
        self.train_mode = train_mode

        # specify whether choosing to save the images in numpy matrix or not
        self.auto_save = auto_save

        # specify whether transformation are needed or not
        self.transform = transform

        with open(self.file_path) as file:
            table = csv.reader(file, delimiter=self.token)
            for row in table:
                short_col = row[0]  # extract path from first column of i-th row, i.e. short exposure image's path
                long_col = row[1]  # extract path second column of i-th row, i.e. long exposure image's path

                if platform == 'win32':
                    short_col = short_col.replace("/", "\\")
                    long_col = long_col.replace("/", "\\")

                # append short and long paths with the root directory, in order to create the final path
                short_path = os.path.join(ROOT_DIR, short_col[2:])
                long_path = os.path.join(ROOT_DIR, long_col[2:])

                self.paths.append((short_path, long_path))

    def _len_(self):
        return len(self.paths)

    def _getitem_(self, index):
        short_path, long_path = self.paths[index]

        print(f'DEBUG _getitem_: {short_path}, {long_path}\tindex: {index}\n')

        short_image_raw = rawpy.imread(short_path)
        long_image_raw = rawpy.imread(long_path)

        # Apply transformations if defined
        if self.transform:
            short_image, long_image = self.transform(short_image_raw, long_image_raw)
        else:
            short_image, long_image = short_image_raw, long_image_raw

        if self.train_mode is True and self.auto_save is True:
            raise Exception("Error")

        if self.train_mode is True:
            segmenta_s(short_image, 512, short_path)
            segmenta_l(long_image, 1024, long_path)

        if self.auto_save is True:
            if not os.path.exists(os.path.join(DATA_DIR, 'numpy_short')):
                os.makedirs(os.path.join(DATA_DIR, 'numpy_short'))
            if not os.path.exists(os.path.join(DATA_DIR, 'numpy_long')):
                os.makedirs(os.path.join(DATA_DIR, 'numpy_long'))

            np.save(short_path.replace('short', 'numpy_short').replace('ARW', 'npy'), short_image)
            np.save(long_path.replace('long', 'numpy_long').replace('ARW', 'npy'), long_image)

        # print(f'DEBUG _getitem_: short {short_image.shape}, long {long_image.shape}')

        return short_image, long_image, short_path, long_path


class ToSidRaw:
    def _call_(self, short_raw, long_raw):
        self.short = self.pack_raw(short_raw)
        self.short = np.transpose(self.short, (2, 0, 1))  # make it channel-first
        # print(f"DEBUG| short exposed image: type {type(self.short)}, dims {self.short.shape}")

        self.long = long_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        self.long = np.float32(self.long / 65535.0)
        # print(f"DEBUG| long exposed image: type {type(self.long)}, dims {self.long.shape}")
        """self.long = long_raw.raw_image_visible.astype(np.float32)
        self.long = np.float32(self.long/65535.0)"""

        return self.short, self.long

    @staticmethod
    def pack_raw(raw):
        """
            Subtract black level: img-512 reduces intensity;
            set negatives values to 0;
            normalize with (image_bit-512)=((2^14-1)-512)
        """
        img = raw.raw_image_visible.astype(np.float32)
        # print(f"DEBUG pack_raw : {type(img)}")

        img = np.maximum(img - 512, 0) / (16383 - 512)

        img = np.expand_dims(img, axis=2)  # get Bayer's matrix (chessboard)

        height = img.shape[0]
        width = img.shape[1]

        out = np.concatenate(
            (img[0:height:2, 0:width:2, :],
             # read only red pixels and put them into first matrix
             img[0:height:2, 1:width:2, :],
             # read only green pixels and put them into second matrix
             img[1:height:2, 1:width:2, :],
             # read only blue pixels and put them into third matrix
             img[1:height:2, 0:width:2, :]),
            # read only the remaining green pixels and put them into fourth matrix
            axis=2)

        return out


class MultiCompose(Compose):

    def _init_(self, transforms):
        super()._init_(transforms)

    def _call_(self, *imgs):
        self.transformed_imgs = []
        for t in self.transforms:
            self.transformed_imgs = t(*imgs)

        return self.transformed_imgs


class Rescale:
    """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
    """

    def _init_(self, output_size):
        self.output_size = output_size

    def _call_(self, img):
        do_nothing = True


# scale = Rescale(512)
to_sid_raw = ToSidRaw()
# normalize = Normalize([...], [...])
# to_tensor = ToTensor(...)

composed = MultiCompose([to_sid_raw])

# train_set = SidDataset(TRAIN_PATHS, token=' ', train_mode=True, transform=composed)
# valid_set = SidDataset(VALID_PATHS, token=' ', auto_save=True, transform=composed)
test_set = SidDataset(TEST_PATHS, token=' ', auto_save=True, transform=composed)

i = 0
size = len(test_set)
for img, gt, img_path, gt_path in test_set:
    print(f'{i} of {size}: img {img_path}, gt {gt_path} saved')
    i += 1