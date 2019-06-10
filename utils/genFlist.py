#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tifffile
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str,  required=True,   help="the original dataset path")
parser.add_argument("--output_dir",    type=str,  default="./",    help="the dir to save flist files")
args = parser.parse_args()


def get_flist(filetype):
    img_list = []
    wildcard_image = '*%s*.%s' % ('_RGB', 'tif')
    glob_paths = glob(os.path.join(args.dataset_dir, wildcard_image))

    for path in tqdm(glob_paths):
        image_name = os.path.split(path)[-1].replace('_RGB', filetype)
        img_list.append(image_name)

    return img_list


def save_flist(flist, file_name):
    out_path = os.path.join(args.output_dir, file_name)
    with open(out_path, 'w') as f:
        for fname in flist:
            f.write(fname + '\n')


def main():
    print("Saving _RGB top image flist..")
    img_flist = get_flist('_RGB')
    save_flist(img_flist, 'dfc_top_all.txt')

    print("Saving _CLS label flist..")
    cls_flist = get_flist('_CLS')
    save_flist(cls_flist, 'dfc_seg_all.txt')

    print("Saving _AGL dsm flist..")
    agl_flist = get_flist('_AGL')
    save_flist(agl_flist, 'dfc_dsm_all.txt')


if __name__ == '__main__':
    main()
