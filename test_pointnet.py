"""
Create: 2022.06.13
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import argparse 

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import PointCloudData
from source.model import PointNet

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        type=str,
                        default="classification")
    parser.add_argument("--weight_path",
                        type=str,
                        default="pointnet.pth")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    root_path = args.root_path

    fold_path = os.path.join(root_path, 'val')
    folders = [dir for dir in sorted(os.listdir(fold_path)) if os.path.isdir(os.path.join(fold_path, dir))]
    classes = {fold: i for i, fold in enumerate(folders)}

    valid_ds = PointCloudData(root_path, valid = True, folder = 'val')
    valid_loader = DataLoader(dataset = valid_ds, batch_size = 16)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pointnet = PointNet()
    pointnet.to(device)

    pointnet.load_state_dict(torch.load(args.weight_path))
    pointnet.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            print('Batch [%4d / %4d]' % (i + 1, len(valid_loader)))

            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            outputs, _, _ = pointnet(inputs.transpose(1, 2))
            _, preds = torch.max(outputs.data, 1)

            all_preds += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

            print('Normalized confusion matrix')
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.figure(figsize = (8, 8))
    plot_confusion_matrix(cm, list(classes.keys()), normalize = False)

    plt.show()