#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tifffile
import numpy as np
from glob import glob
from PIL import Image
from .trans import convert_labels

# for data augmentation
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness
)


class DFCloader(object):
    """
    param data_path: the DFC data folder
    param mode: data type to load, select from ['train_data', 'test_data', 'all_data']
    param img_height: the height of cropped image
    param img_width: the width of cropped image
    param block_height: the height of cropped block
    param block_width: the width of cropped block
    param block_overlap: the cropped overlap between blocks
    """

    def __init__(self,
                 data_path,
                 mode,
                 img_height=1024,
                 img_width=1024,
                 block_height=512,
                 block_width=512,
                 block_overlap=256):

        self.img_height = img_height
        self.img_width = img_width
        self.data_path = data_path
        self.block_height = block_height
        self.block_width = block_width
        self.block_overlap = block_overlap
        self.mode = mode
        self.top_data = []
        self.dsm_data = []
        self.cls_data = []

        dir_path = os.path.dirname(os.path.realpath(__file__))
        top_train_file = dir_path + '/filenames/dfc_top_train' + '.txt'
        dsm_train_file = dir_path + '/filenames/dfc_agl_train' + '.txt'
        cls_train_file = dir_path + '/filenames/dfc_cls_train' + '.txt'
        top_test_file = dir_path + '/filenames/dfc_top_test' + '.txt'
        dsm_test_file = dir_path + '/filenames/dfc_agl_test' + '.txt'
        cls_test_file = dir_path + '/filenames/dfc_cls_test' + '.txt'

        with open(top_train_file, 'r') as f:
            self.top_train_flist = [i[:-1] for i in f.readlines()]
        with open(dsm_train_file, 'r') as f:
            self.dsm_train_flist = [i[:-1] for i in f.readlines()]
        with open(cls_train_file, 'r') as f:
            self.cls_train_flist = [i[:-1] for i in f.readlines()]
        with open(top_test_file, 'r') as f:
            self.top_test_flist = [i[:-1] for i in f.readlines()]
        with open(dsm_test_file, 'r') as f:
            self.dsm_test_flist = [i[:-1] for i in f.readlines()]
        with open(cls_test_file, 'r') as f:
            self.cls_test_flist = [i[:-1] for i in f.readlines()]

        self.top_all_flist = self.top_train_flist + self.top_test_flist
        self.dsm_all_flist = self.dsm_train_flist + self.dsm_test_flist
        self.cls_all_flist = self.cls_train_flist + self.cls_test_flist

    def get_image_paths(self, flist):
        img_paths = [os.path.join(self.data_path, filename) for filename in flist]
        return img_paths

    def get_train_image_paths(self):
        return self.get_image_paths(self.top_train_flist), \
               self.get_image_paths(self.dsm_train_flist), \
               self.get_image_paths(self.cls_train_flist)

    def get_test_image_paths(self):
        return self.get_image_paths(self.top_test_flist), \
               self.get_image_paths(self.dsm_test_flist), \
               self.get_image_paths(self.cls_test_flist)

    def get_all_image_paths(self):
        return self.get_image_paths(self.top_all_flist), \
               self.get_image_paths(self.dsm_all_flist), \
               self.get_image_paths(self.cls_all_flist)

    def load_image_data(self, file_path, mode):
        # load tiff image as PIL image
        img = Image.open(file_path)
        image_pil = Image.frombytes(mode, img.size, img.tobytes())
        img_arr = np.asarray(image_pil)
        return Image.fromarray(img_arr[0:self.img_height, 0:self.img_width])

    def load_tiff_image_as_np(self, file_path):
        # load any tiff as float-32 np array
        img_arr = tifffile.imread(file_path).astype(np.float32)
        return img_arr[0:self.img_height, 0:self.img_width]

    def load_float_image_as_pil(self, filename):
        # load tiff image(float-32 gray) as an PIL image
        return self.load_image_data(filename, 'F')

    def load_RGB_image_as_pil(self, filename):
        # load tiff(RGB-8 sRGB) as an PIL image
        return self.load_image_data(filename, 'RGB')

    def image_augmentation(self, currImg, gt):
        aug = Compose([VerticalFlip(p=0.5),
                       RandomRotate90(p=0.5),
                       HorizontalFlip(p=0.5),
                       Transpose(p=0.5)])

        augmented = aug(image=currImg, mask=gt)
        imageMedium = augmented['image']
        labelMedium = augmented['mask']
        return imageMedium, labelMedium

    def get_blocks(self):
        blocks = []
        yEnd = self.img_height - self.block_height
        xEnd = self.img_width - self.block_width
        x = np.linspace(0, xEnd, np.ceil(xEnd/np.float(self.block_width-self.block_overlap))+1, endpoint=True).astype('int')
        y = np.linspace(0, yEnd, np.ceil(yEnd/np.float(self.block_height-self.block_overlap))+1, endpoint=True).astype('int')

        for curry in y:
            for currx in x:
                blocks.append((curry, currx))

        return blocks

    def get_data(self, imgPaths):
        data = []

        if (self.block_height == 0) and (self.block_width == 0):
            for imgPath in imgPaths:
                data.append((imgPath, 0, 0))
        else:
            blocks = self.get_blocks()
            for imgPath in imgPaths:
                for block in blocks:
                    data.append((imgPath, block[0], block[1]))

        return data

    def get_pair_data(self):
        if not self.mode in ['train_data', 'test_data', 'all_data']:
            raise ValueError("Panic::Unvalid mode")

        if self.mode == 'train_data':
            top_paths, dsm_paths, cls_paths = self.get_train_image_paths()
        elif self.mode == 'test_data':
            top_paths, dsm_paths, cls_paths = self.get_test_image_paths()
        elif self.mode == 'all_data':
            top_paths, dsm_paths, cls_paths = self.get_all_image_paths()
        else:
            pass

        self.top_data = self.get_data(top_paths)
        self.dsm_data = self.get_data(dsm_paths)
        self.cls_data = self.get_data(cls_paths)

    def get_data_length(self):
        if (self.top_data == []) and (self.dsm_data == []) and (self.cls_data == []):
            self.get_pair_data()
        else:
            pass

        return len(self.top_data)

    @staticmethod
    def to_categorical(y, num_classes=None, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape

        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])

        y = y.ravel()

        if not num_classes:
            num_classes = np.max(y) + 1

        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def load_item(self, idx, gtType="dsm"):
        if (self.top_data == []) and (self.dsm_data == []) and (self.cls_data == []):
            self.get_pair_data()
        else:
            pass

        idx_bound = len(self.top_data)
        try:
            assert idx_bound == len(self.dsm_data)
        except AssertionError:
            print("AssertionError: Length of 'top_data' and 'dsm_data' are not equal.")
            print("Length of top_data: {}, length of dsm_data: {}".format(idx_bound, len(self.dsm_data)))

        if idx >= idx_bound:
            raise IndexError("The index of data should less than {}, given {}.".format(idx_bound, idx))
        else:
            pass

        curr_imgData = self.top_data[idx]
        curr_dsmData = self.dsm_data[idx]
        curr_clsData = self.cls_data[idx]

        currImg = self.load_tiff_image_as_np(curr_imgData[0])
        if gtType == "dsm":
            currGt = self.load_tiff_image_as_np(curr_dsmData[0])
        elif gtType == "cls":
            currGt = self.load_tiff_image_as_np(curr_clsData[0])
            currGt = convert_labels(currGt, toLasStandard=False)
        else:
            pass

        # deal with block if it is set
        rStart, cStart = curr_imgData[1:3]

        if (self.block_height == 0) and (self.block_width == 0):
            rEnd, cEnd = (rStart+self.img_height, cStart+self.img_width)
        else:
            rEnd, cEnd = (rStart+self.block_height, cStart+self.block_width)

        currImg = currImg[rStart:rEnd, cStart:cEnd, :]
        currGt = currGt[rStart:rEnd, cStart:cEnd]

        # data augmentation
        if self.mode != "test_data":
            imageMedium, labelMedium = self.image_augmentation(currImg, currGt)
        else:
            imageMedium, labelMedium = (currImg, currGt)

        return imageMedium, labelMedium
