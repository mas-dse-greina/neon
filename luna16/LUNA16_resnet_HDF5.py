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
python LUNA16_resnet_HDF5.py -z 256 -b gpu -i 0 -e 40 --depth 50 --subset 0

NOTE: A batch size of 256 seems to work the best. Larger batch sizes train too slowly.
If you opt for larger batch size, then you might need to increase the learning rate.

"""
import itertools as itt
from neon import logger as neon_logger

from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend

from neon.optimizers import Adam, Schedule, MultiOptimizer, RMSProp
from neon.transforms import Cost, Softmax
from neon.transforms import Rectlin, CrossEntropyBinary
from neon.models import Model

from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation, Dropout
from neon.layers import MergeSum, SkipNode, BatchNorm

from neon.data import HDF5IteratorOneHot

from neon.util.persist import load_obj

import os
import numpy as np


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

# valid_set = HDF5IteratorOneHot('/mnt/data/medical/luna16/luna16_roi_subset{}_ALL.h5'.format(SUBSET), \
#                                 flip_enable=False, rot90_enable=False, crop_enable=False, border_size=5)

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

    layers = [Conv(name='Input Layer', **conv_params(7, 64, strides=2)),
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

    layers.append(Pooling('all', op='avg', name='end_resnet'))
    layers.append(Conv(name = 'Custom Head 1', **conv_params(1, 1000, relu=True))) 
    layers.append(Dropout(0.5))
    layers.append(Conv(name = 'Custom Head 2', **conv_params(1, 2, relu=False))) 
    layers.append(Activation(Softmax()))
    # layers.append(Affine(512, init=Kaiming(local=False), 
    #                  batch_norm=True, activation=Rectlin()))
    # layers.append(Affine(2, init=Kaiming(local=False), activation=Softmax()))

    return Model(layers=layers)

lunaModel = create_network(args.depth)

PRETRAINED = False
# Pre-trained ResNet 50 
# It assumes the image has a depth channel of 3
pretrained_weights_file = 'resnet{}_weights.prm'.format(args.depth)
print ('Loading pre-trained ResNet weights: {}'.format(pretrained_weights_file))
trained_resnet = load_obj(pretrained_weights_file)  # Load a pre-trained resnet 50 model

# Load the pre-trained weights to our model
param_layers = [l for l in lunaModel.layers.layers]
param_dict_list = trained_resnet['model']['config']['layers']

for layer, params in zip(param_layers, param_dict_list):

    if (layer.name == 'end_resnet'):
        break

    # ResNet is trained on images that have 3 color depth channels
    # Our data usually isn't 3 color channels so we should not load the weights for that layer
    if (layer.name != 'Input Layer'):
        layer.load_weights(params, load_states=False)  # Don't load the state, just load the weights

del trained_resnet
PRETRAINED = True
print('Pre-trained weights loaded.')

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

modelFileName = 'LUNA16_resnetHDF_subset{}.prm'.format(SUBSET)

# #If model file exists, then load the it and start from there.
# if (os.path.isfile(modelFileName)):
#   lunaModel = Model(modelFileName)

optHead = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)


if PRETRAINED:
    optPretrained = Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999) # Set a slow learning rate for ResNet layers
else:
    optPretrained = optHead


mapping = {'default': optPretrained, # default optimizer applied to the pretrained sections
           'Input Layer' : optHead, # The layer named 'Input Layer'
           'Custom Head 1' : optHead,
           'Custom Head 2' : optHead,
           'Affine': optHead} # all layers from the Affine class

# use multiple optimizers
opt = MultiOptimizer(mapping)

# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
    
# configure callbacks
callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
# add a callback that saves the best model state
callbacks.add_save_best_state_callback(modelFileName)


lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)


