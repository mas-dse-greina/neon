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
ResNet with three merged layers from the 2D convolutions of 
transverse, sagittal, and coronal planes.

Command:
python LUNA16_resnet_3D_merge.py -z 128 -e 200 -b gpu -i 0

"""

from neon import logger as neon_logger
from neon.initializers import Gaussian, GlorotUniform, Xavier, Constant
from neon.optimizers import Adam, Adadelta
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost, Affine, MergeMultistream, Sequential
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Logistic, CrossEntropyBinary, Misclassification, PrecisionRecall
from neon.models import Model
from aeon import DataLoader
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from neon.data.dataloader_transformers import BGRMeanSubtract, TypeCast, OneHot
import numpy as np

from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation, Dropout
from neon.layers import MergeSum, SkipNode, BatchNorm

from neon.data.datasets import Dataset
from neon.util.persist import load_obj
import os

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=5,
                    help='depth of each stage (network depth will be 9n+2)')

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

# Set up the training set to load via aeon
# Augmentating the data via flipping, rotating, changing contrast/brightness
image_config = dict(height=64, width=64, flip_enable=True, channels=3,
                    contrast=(0.9,1.1), brightness=(0.9,1.1), 
                    scale=(0.75,0.75), fixed_aspect_ratio=True)
label_config = dict(binary=False)


config = dict(type="image,image,image,label",
                image=image_config,
                label=label_config,
                manifest_filename='manifest_all_but_9_planes.csv',
                minibatch_size=args.batch_size,
                macrobatch_size=128,
                cache_directory='cache_dir',
                shuffle_manifest=True)
                #shuffle_every_epoch = True)

dataset = DataLoader(config, be)
train_set = TypeCast(dataset, index=0, dtype=np.float32) # cast image to float

# Set up the validation set to load via aeon
image_config = dict(height=64, width=64, channels=3)
label_config = dict(binary=False)

config = dict(type="image,image,image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset9_augmented_planes.csv',
              minibatch_size=args.batch_size,
              macrobatch_size=128,
              cache_directory='cache_dir',
              shuffle_manifest=True)
              #shuffle_every_epoch = True)
dataset = DataLoader(config, be)
valid_set = TypeCast(dataset, index=0, dtype=np.float32) # cast image to float

def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=batch_norm)


def module_s1(nfm, first=False):
    '''
    non-strided
    '''
    sidepath = Conv(**conv_params(1, nfm * 4, 1, False, False)) if first else SkipNode()
    mainpath = [] if first else [BatchNorm(), Activation(Rectlin())]
    mainpath.append(Conv(**conv_params(1, nfm)))
    mainpath.append(Conv(**conv_params(3, nfm)))
    mainpath.append(Conv(**conv_params(1, nfm * 4, relu=False, batch_norm=False)))

    return MergeSum([sidepath, mainpath])


def module_s2(nfm):
    '''
    strided
    '''
    module = [BatchNorm(), Activation(Rectlin())]
    mainpath = [Conv(**conv_params(1, nfm, stride=2)),
                Conv(**conv_params(3, nfm)),
                Conv(**conv_params(1, nfm * 4, relu=False, batch_norm=False))]
    sidepath = [Conv(**conv_params(1, nfm * 4, stride=2, relu=False, batch_norm=False))]
    module.append(MergeSum([sidepath, mainpath]))
    return module


def create_layers(stage_depth):
    # Structure of the deep residual part of the network:
    # stage_depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
    nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * stage_depth)]
    strides = [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

    # Now construct the network
    layers = [Conv(**conv_params(3, 16))]
    layers.append(module_s1(nfms[0], True))

    for nfm, stride in zip(nfms[1:], strides):
        res_module = module_s1(nfm) if stride == 1 else module_s2(nfm)
        layers.append(res_module)
    # layers.append(BatchNorm())
    # layers.append(Activation(Rectlin()))
    # layers.append(Pooling('all', op='avg'))
    # layers.append(Affine(1, init=Kaiming(local=False), batch_norm=True, activation=Logistic()))

    return layers

opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0005, schedule=Schedule([40, 70], 0.1))

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# Create three separate streams: transverse, coronal, and sagittal 2D planes
lunaModel_trans = Sequential(create_layers(args.depth)) # transverse plane

lunaModel_sag = Sequential(create_layers(args.depth))   # sagittal plane

lunaModel_cor = Sequential(create_layers(args.depth))   # coronal plane

# Merge all 3 streams together.
layers = [MergeMultistream(layers=[lunaModel_trans, lunaModel_sag, lunaModel_cor], merge='stack')]

layers.append(BatchNorm())
layers.append(Activation(Rectlin()))
layers.append(Pooling('all', op='avg'))
layers.append(Affine(1, init=Kaiming(local=False), batch_norm=True, activation=Logistic()))





# initialize model
path1 = Sequential(layers=[Affine(nout=100, init=Kaiming(local=False), activation=Rectlin()),
                           Affine(nout=100, init=Kaiming(local=False), activation=Rectlin())])

path2 = Sequential(layers=[Affine(nout=100, init=Kaiming(local=False), activation=Rectlin()),
                           Affine(nout=100, init=Kaiming(local=False), activation=Rectlin())])

layers = [MergeMultistream(layers=[path1, path2, path1], merge="stack"),
          Affine(nout=10, init=Kaiming(local=False), activation=Logistic(shortcut=True))]



lunaModel = Model(layers=layers)

# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
    
# configure callbacks
callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
# add a callback that saves the best model state
callbacks.add_save_best_state_callback('LUNA16_resnet_3D_merge.prm')


lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

lunaModel.save_params('LUNA16_resnet_3D_merge.prm')


