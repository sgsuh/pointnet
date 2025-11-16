"""
Create: 2022.06.13
Author: SG.SUH
Python: 3.7
"""

import os
import glob
import shutil
import random
import argparse 

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_root_fold",
                        type=str,
                        default="training/save")
    parser.add_argument("--dst_root_fold",
                        type=str,
                        default="classification")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    src_root_fold = args.src_root_fold
    dst_root_fold = args.dst_root_fold
    dst_train_fold = os.path.join(dst_root_fold, 'train')
    dst_val_fold = os.path.join(dst_root_fold, 'val')

    if not os.path.isdir(dst_train_fold):
        os.makedirs(dst_train_fold)

    if not os.path.isdir(dst_val_fold):
        os.makedirs(dst_val_fold)

    src_fold_list = glob.glob(os.path.join(src_root_fold, '*'))
    src_fold_list.sort()

    for src_fold_path in src_fold_list:
        fold_name = os.path.basename(src_fold_path)

        src_file_list = glob.glob(os.path.join(src_fold_path, '*.npy'))
        src_file_list.sort()

        random.shuffle(src_file_list)

        train_num = int(len(src_file_list) * 0.8)
        val_num = len(src_file_list) - train_num

        for i, src_file_path in enumerate(src_file_list):
            file_name = os.path.basename(src_file_path)

            if i < train_num:
                dst_fold_path = os.path.join(dst_train_fold, fold_name)            
            else:
                dst_fold_path = os.path.join(dst_val_fold, fold_name)

            if not os.path.isdir(dst_fold_path):
                os.makedirs(dst_fold_path)

            dst_file_path = os.path.join(dst_fold_path, file_name)

            print(dst_file_path)

            shutil.copy(src_file_path, dst_file_path)        
