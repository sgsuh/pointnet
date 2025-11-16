"""
Create: 2022.06.13
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import numpy as np
import torch
import os
import random
import math

from torch.utils.data import Dataset
from torchvision import transforms

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)

        self.output_size = output_size

    def __call__(self, points):
        total_point = points.shape[0]

        rand_idx = np.random.choice(total_point, size = self.output_size, replace = False) 

        sample_points = points[rand_idx]

        return sample_points

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis = 0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis = 1))

        return norm_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([PointSampler(128), Normalize(), ToTensor()])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid = False, folder = 'train', transform = default_transforms()):
        self.root_dir = root_dir
        self.fold_path = os.path.join(self.root_dir, folder)
        self.valid = valid
        self.transforms = transform if not valid else default_transforms()
        self.files = []

        folders = [dir for dir in sorted(os.listdir(self.fold_path)) if os.path.isdir(os.path.join(self.fold_path, dir))]
        self.classes = {fold: i for i, fold in enumerate(folders)}

        for category in self.classes.keys():
            class_fold = os.path.join(self.fold_path, category)

            for file_path in os.listdir(class_fold):
                sample = {}

                sample['path'] = os.path.join(class_fold, file_path)
                sample['category'] = category

                self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, points):
        if self.transforms:
            points = self.transforms(points)

        return points

    def __getitem__(self, idx):
        file_path = self.files[idx]['path']
        category = self.files[idx]['category']

        points = np.load(file_path)
        pointcloud = self.__preproc__(points[:, :3])

        return {'pointcloud': pointcloud, 'category': self.classes[category]}

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T

        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise

        return noisy_pointcloud