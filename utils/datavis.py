#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tifffile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str,  required=True,        help="the original dataset path")
parser.add_argument("--output_dir",    type=str,  default="./gtShow",   help="the dir to save shown gt images")
parser.add_argument("--cmap_method" ,  type=str,  default="jet",        help="the color method for cmap")
parser.add_argument("--check_mode",    type=str,  default="vis",        help="visual or statistics: ['vis', 'sta']")
parser.add_argument("--sta_mode",      type=str,  default="high",       help="select from: ['total', 'high']")
args = parser.parse_args()


def read_img(fpath):
    img_arr = tifffile.imread(fpath)
    return img_arr


def get_flist(name_file_path):
    with open(name_file_path, 'r') as f:
        flist = [name[:-1] for name in f.readlines()]
    return flist


def save_img(img_arr, outname):
    save_path = os.path.join(args.output_dir, outname)
    plt.imsave(save_path, img_arr, cmap=args.cmap_method)


def save_gt(flist):
    for fname in tqdm(flist):
        fpath = os.path.join(args.dataset_dir, fname)
        outname = "shown_" + fname
        try:
            img_arr = read_img(fpath)
            save_img(img_arr, outname)
        except IOError:
            print("cannot open {}".format(fpath))


def study_gt(flist):
    flen = len(flist)
    glob_arr = np.zeros([flen, 1024, 1024]).astype(np.float16)

    for it, fname in enumerate(tqdm(flist)):
        fpath = os.path.join(args.dataset_dir, fname)
        glob_arr[it] = np.nan_to_num(read_img(fpath))

    glob_arr = glob_arr.flatten()

    if args.sta_mode == 'high':
        glob_arr = glob_arr[glob_arr >= 2.]

    print("Getting data info ...")
    glob_min, glob_max, glob_mean = map(lambda arr: np.round(arr, decimals=2), [np.min(glob_arr), np.max(glob_arr), np.mean(glob_arr)])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    n, bins, patches = plt.hist(x=glob_arr, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.85)
    plt.xlabel('Height range')
    plt.ylabel('Frequency')

    if args.sta_mode == 'total':
        plt.title("Histogram of DFC $\\rightarrow$ $\mu:$ {:.2f}, min: {:.2f}, max: {:.2f}".format(glob_mean, glob_min, glob_max))
    elif args.sta_mode == 'high':
        plt.title("DFC Height $(>2)$ $\\rightarrow$ $\mu:$ {:.2f}, min: {:.2f}, max: {:.2f}".format(glob_mean, glob_min, glob_max))
    else:
        pass

    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    if args.sta_mode == 'total':
        plt.savefig(os.path.join(args.output_dir, 'dfc_info.png'))
    elif args.sta_mode == 'high':
        plt.savefig(os.path.join(args.output_dir, 'dfc_info_high.png'))
    else:
        pass


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    name_file_path = "dfc_dsm_all.txt"

    flist = get_flist(name_file_path)

    if args.check_mode == 'vis':
        print("Saving colorized ground truth ...")
        save_gt(flist)
    elif args.check_mode == 'sta':
        print("Checking ...")
        study_gt(flist)
    else:
        raise ValueError("Panic: Unvalid check mode")


if __name__ == '__main__':
    main()
