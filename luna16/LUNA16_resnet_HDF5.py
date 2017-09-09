#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
ResNet on LUNA16 data.

Depth parameter can be 18, 34, 50, 101, or 152

Command:
python LUNA16_resnet_HDF5.py -z 1024 -b gpu -i 0 --depth 50 -e 40 --subset 9 

"""
import itertools as itt
from neon import logger as neon_logger

from neon.optimizers import Adam, Adadelta
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.transforms import Cost, Softmax
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, LogLoss, Misclassification, PrecisionRecall
from neon.models import Model
from aeon import DataLoader
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from neon.data.dataloader_transformers import TypeCast
import numpy as np

from neon.initializers import Kaiming, IdentityInit, Constant
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation, Dropout
from neon.layers import MergeSum, SkipNode, BatchNorm

from neon.data.datasets import Dataset
from neon.util.persist import load_obj
import os

from neon.data import HDF5IteratorOneHot


# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=50,
                    help='choices 18, 34, 50, 101, 152')
parser.add_argument('--subset', type=int, default=9)

args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs

# Next line gets rid of the deterministic warning
args.deterministic = None

if (args.rng_seed is None):
  args.rng_seed = 16

print('Batch size = {}'.format(args.batch_size))

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
#be.enable_winograd = 4  # default to winograd 4 for fast autotune

SUBSET = args.subset
train_set = HDF5IteratorOneHot('/mnt/data/medical/luna16/luna16_roi_except_subset{}_augmented.h5'.format(SUBSET), \
                                 flip_enable=True, rot90_enable=True, crop_enable=False, border_size=5)


valid_set = HDF5IteratorOneHot('/mnt/data/medical/luna16/luna16_roi_subset{}_augmented.h5'.format(SUBSET), \
                                flip_enable=False, rot90_enable=False, crop_enable=False, border_size=5)

print('Using subset{}'.format(SUBSET))

'''
ResNet Model
Taken from https://github.com/NervanaSystems/neon/blob/master/examples/imagenet/network_msra.py
'''

# A resnet module
#
#             - mainpath = Convolution - 
#           /                            \
# input   -                                Sum - Relu - output
#           \                            /
#            -   sidepath = Identity   - 
#
# For networks > 50 layers, we'll use the updated bottleneck
# which just moves the Relu into the mainpath so that
# the sidepath is a clean Indentity function.
# https://arxiv.org/pdf/1603.05027.pdf
#
#             - mainpath = Convolution - Relu - 
#           /                                   \
# input   -                                       Sum - output
#           \                                   /
#            -       sidepath = Identity       - 

def conv_params(fsize, nfm, strides=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm),
                strides=strides,
                activation=(Rectlin() if relu else None),
                padding=(fsize // 2),
                batch_norm=batch_norm,
                init=Kaiming(local=True))


def module_factory(nfm, bottleneck=True, stride=1):
    nfm_out = nfm * 4 if bottleneck else nfm
    use_skip = True if stride == 1 else False
    stride = abs(stride)
    sidepath = [SkipNode() if use_skip else Conv(
        **conv_params(1, nfm_out, stride, False))]

    if bottleneck:
        mainpath = [Conv(**conv_params(1, nfm, stride)),
                    Conv(**conv_params(3, nfm)),
                    Conv(**conv_params(1, nfm_out, relu=False))]
    else:
        mainpath = [Conv(**conv_params(3, nfm, stride)),
                    Conv(**conv_params(3, nfm, relu=False))]
    return [MergeSum([mainpath, sidepath]),
            Activation(Rectlin())]


def create_network(stage_depth):
    if stage_depth in (18, 18):
        stages = (2, 2, 2, 2)
    elif stage_depth in (34, 50):
        stages = (3, 4, 6, 3)
    elif stage_depth in (68, 101):
        stages = (3, 4, 23, 3)
    elif stage_depth in (102, 152):
        stages = (3, 8, 36, 3)
    else:
        raise ValueError('Invalid stage_depth value'.format(stage_depth))

    bottleneck = False
    if stage_depth in (50, 101, 152):
        bottleneck = True

    layers = [Conv(**conv_params(7, 64, strides=2)),
              Pooling(3, strides=2)]

    # Structure of the deep residual part of the network:
    # stage_depth modules of 2 convolutional layers each at feature map depths
    # of 64, 128, 256, 512
    nfms = list(itt.chain.from_iterable(
        [itt.repeat(2**(x + 6), r) for x, r in enumerate(stages)]))
    strides = [-1] + [1 if cur == prev else 2 for cur,
                      prev in zip(nfms[1:], nfms[:-1])]

    for nfm, stride in zip(nfms, strides):
        layers.append(module_factory(nfm, bottleneck, stride))

    layers.append(Pooling('all', op='avg'))
    # layers.append(Conv(**conv_params(1, 10, relu=True))) 
    # layers.append(Conv(**conv_params(1, 2, relu=False))) 
    layers.append(Affine(10, init=Kaiming(local=False), 
                     batch_norm=True, activation=Rectlin()))
    layers.append(Affine(2, init=Kaiming(local=False), activation=Softmax()))

    return Model(layers=layers)

lunaModel = create_network(args.depth)

cost = GeneralizedCost(costfunc=CrossEntropyBinary(weight=0.5))

modelFileName = 'LUNA16_resnetHDF_subset{}.prm'.format(SUBSET)

# If model file exists, then load the it and start from there.
# if (os.path.isfile(modelFileName)):
#   lunaModel = Model(modelFileName)


weight_sched = Schedule([5, 8], 0.1)
#opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=weight_sched)
opt = Adadelta(decay=0.95, epsilon=1e-6) 

# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
    
# configure callbacks
callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
# add a callback that saves the best model state
callbacks.add_save_best_state_callback(modelFileName)


lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)


