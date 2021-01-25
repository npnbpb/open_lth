# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision
import numpy as np

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """The SVHN dataset."""

    @staticmethod
    def num_train_examples(): return 73257

    @staticmethod
    def num_test_examples(): return 26032

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        # No augmentation for SVHN.
        train_set = torchvision.datasets.SVHN(
            split="train", root=os.path.join(get_platform().dataset_root, 'svhn'), download=True)
        train_set.data = np.swapaxes(train_set.data, 1,3)
        return Dataset(train_set.data, train_set.labels)

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.SVHN(
            split="test", root=os.path.join(get_platform().dataset_root, 'svhn'), download=True)

        test_set.data = np.swapaxes(test_set.data, 1,3)
        return Dataset(test_set.data, test_set.labels)

    def __init__(self,  examples, labels):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example, mode='RGB')


DataLoader = base.DataLoader
