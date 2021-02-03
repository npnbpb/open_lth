# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
from math import floor


class Model(base.Model):
    '''A simple CNN for MNIST'''

    def __init__(self, channels, initializer, outputs=10):
        super(Model, self).__init__()

        mnist_size = 784  # 28 * 28 = number of pixels in MNIST image.
        
        layers = []
        in_features = 1
        for out_features in channels:
            layers.append(nn.Conv2d(in_features, out_features, 3, 1, 1))
            in_features = out_features


        final_size = int(floor(28/(2**len(channels)))**2 * out_features)
        self.conv_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(final_size, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        
        for layer in self.conv_layers:
            x = F.relu(layer(x))
            x = F.avg_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)  # Flatten.
        return self.fc(x)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('mnist_simplecnn') and
                len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is mnist_simplecnn_N1[_N2...].

        N1, N2, etc. are the number of features in each convolutional layer excluding the
        output layer (10 neurons by default). A simplecnn with 300 features in the first conv layer,
        100 feature in the second conv layer, and 10 output neurons is 'mnist_simplecnn_300_100'.
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        plan = [int(n) for n in model_name.split('_')[2:]]
        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_simplecnn_16_32',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=5e-2,
            training_steps='3ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
