#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import decimal
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from decimal import Decimal


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str,    required=True,           help="the original dataset path")
parser.add_argument("--output_dir",    type=str,    default="./filenames",   help="the dir to save out files")
parser.add_argument("--test_ratio",    type=float,  default=0.2,             help="the test dataset ratio")
args = parser.parse_args()


class DataSpliter(object):
    def __init__(self,
                 data_path,
                 test_ratio=0.2):
        self.data_path = data_path
        self.test_ratio = test_ratio

        self.oma_top_wildcard = "OMA_*_RGB.tif"
        self.jax_top_wildcard = "JAX_*_RGB.tif"

        self.oma_top_flist = []
        self.jax_top_flist = []

        self.jax_prefix = []
        self.oma_prefix = []
        self.jax_items = 0
        self.oma_items = 0
        self.total_items = 0

        self.jax_test_prefix = []
        self.oma_test_prefix = []
        self.jax_train_prefix = []
        self.oma_train_prefix = []

    def extract_flist(self, file_paths):
        return list(map(lambda path: os.path.split(path)[-1], file_paths))

    def gen_unique_prefix(self, mode):
        if mode == "OMA":
            top_flist = self.oma_top_flist
        elif mode == "JAX":
            top_flist = self.jax_top_flist
        else:
            raise ValueError("Panic: Unvalid mode name!")

        tmp_list = []
        for fname in top_flist:
            tmp_list.append(fname[:7])

        sorted_unique_list = sorted(list(set(tmp_list)))
        return sorted_unique_list, len(sorted_unique_list)

    def check_items(self):
        self.oma_top_flist = self.extract_flist(glob(os.path.join(self.data_path, self.oma_top_wildcard)))
        self.jax_top_flist = self.extract_flist(glob(os.path.join(self.data_path, self.jax_top_wildcard)))
        self.jax_prefix, self.jax_items = self.gen_unique_prefix("JAX")
        self.oma_prefix, self.oma_items = self.gen_unique_prefix("OMA")
        self.total_items = self.jax_items + self.oma_items

    def round_item(self, item):
        res = Decimal(str(item)).quantize(Decimal('0'), rounding=decimal.ROUND_HALF_UP)
        return int(res)

    def gen_random_items(self, items, clip):
        s = list(range(items))
        random.shuffle(s)
        return s[0:clip]

    def gen_split_prefix(self):
        test_ratio = self.test_ratio

        if(self.total_items == 0):
            self.check_items()
        elif(self.total_items < 0):
            raise ValueError("Panic: the total item appears to be negative!")
        else:
            pass

        test_items = self.round_item(self.total_items * test_ratio)
        train_items = self.total_items - test_items

        try:
            assert self.total_items == test_items + train_items
        except AssertionError:
            print("AssertionError: test items split error!")

        jax_test_items = self.round_item(test_items / self.total_items * self.jax_items)
        oma_test_items = test_items - jax_test_items

        for idx in self.gen_random_items(self.jax_items, jax_test_items):
            self.jax_test_prefix.append(self.jax_prefix[idx])

        for idx in self.gen_random_items(self.oma_items, oma_test_items):
            self.oma_test_prefix.append(self.oma_prefix[idx])

        self.jax_train_prefix = list(set(self.jax_prefix) - set(self.jax_test_prefix))
        self.oma_train_prefix = list(set(self.oma_prefix) - set(self.oma_test_prefix))

    def gen_tops_from_prefix(self, prefix_list):
        res = []
        for prefix in prefix_list:
            item_list = self.extract_flist(glob(os.path.join(self.data_path, (prefix + "_*_RGB.tif"))))
            res.extend(item_list)

        return res


    def top_to_suffix(self, flist, suffix):
        res = []
        for fname in flist:
            image_name = fname.replace('_RGB', suffix)
            res.append(image_name)

        return res

    def save_flist(self, flist, output_dir, file_name):
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, 'w') as f:
            for fname in flist:
                f.write(fname + '\n')

    def save_split_filenames(self, output_dir):
        # initialization
        self.gen_split_prefix()
        oma_top_train = self.gen_tops_from_prefix(self.oma_train_prefix)
        oma_top_test = self.gen_tops_from_prefix(self.oma_test_prefix)
        jax_top_train = self.gen_tops_from_prefix(self.jax_train_prefix)
        jax_top_test = self.gen_tops_from_prefix(self.jax_test_prefix)

        top_train = oma_top_train + jax_top_train
        top_test  = oma_top_test + jax_top_test

        cls_train = self.top_to_suffix(top_train, '_CLS')
        cls_test = self.top_to_suffix(top_test, '_CLS')

        agl_train = self.top_to_suffix(top_train, '_AGL')
        agl_test = self.top_to_suffix(top_test, '_AGL')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save training filenames
        self.save_flist(top_train, output_dir, 'dfc_top_train.txt')
        self.save_flist(cls_train, output_dir, 'dfc_cls_train.txt')
        self.save_flist(agl_train, output_dir, 'dfc_agl_train.txt')

        # save test filenames
        self.save_flist(top_test, output_dir, 'dfc_top_test.txt')
        self.save_flist(cls_test, output_dir, 'dfc_cls_test.txt')
        self.save_flist(agl_test, output_dir, 'dfc_agl_test.txt')


if __name__ == '__main__':
    spliter = DataSpliter(args.dataset_dir, args.test_ratio)
    spliter.save_split_filenames(args.output_dir)
