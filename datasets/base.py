from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from .transforms import create_train_transform, create_test_transform


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for dataset with support for offline distillation.
    """

    def __init__(self, split):
        if split == 'train':
            transforms = create_train_transform()
        else:
            transforms = create_test_transform()
        self.primary_tfl, self.secondary_tfl, self.final_tfl = transforms

        self.features = None

    @abstractmethod
    def _get_data(self, index):
        """
        Returns the image and its label at index.
        """
        pass

    def __getitem__(self, index):
        img, label = self._get_data(index)
        if self.features:
            features = [torch.from_numpy(f[index].copy()) for f in self.features]
            for t in self.primary_tfl:
                img, features = t(img, features)
        else:
            img = self.primary_tfl(img)
            features = []

        img = self.secondary_tfl(img)
        img = self.final_tfl(img)

        return img, label, features
