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

Command:
python LUNA16_resnet.py -z 128 -e 200 -b gpu -i 0



"""

from neon import logger as neon_logger

from neon.optimizers import Adam, Adadelta
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost, Affine
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Logistic, CrossEntropyBinary, Misclassification, PrecisionRecall
from neon.models import Model
from aeon import DataLoader
from neon.callbacks.callbacks import Callbacks
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

patch_size = 32

# Set up the training set to load via aeon
# Augmentating the data via flipping, rotating, changing contrast/brightness
image_config = dict(height=patch_size, width=patch_size, flip_enable=True, channels=3,
                    contrast=(0.9,1.1), brightness=(0.9,1.1), 
                    scale=(0.75,0.75), fixed_aspect_ratio=True)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_all_but_9_trans.csv',
              minibatch_size=args.batch_size,
              macrobatch_size=128,
              cache_directory='cache_dir',
              shuffle_manifest=True)
              #shuffle_every_epoch = True)
train_set = DataLoader(config, be)
train_set = TypeCast(train_set, index=0, dtype=np.float32)  # cast image to float

# Set up the validation set to load via aeon
image_config = dict(height=patch_size, width=patch_size, channels=3)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset9_augmented_trans.csv',
              minibatch_size=args.batch_size)
valid_set = DataLoader(config, be)
valid_set = TypeCast(valid_set, index=0, dtype=np.float32)  # cast image to float


# Set up the testset to load via aeon
# image_config = dict(height=patch_size, width=patch_size, channels=3)
# label_config = dict(binary=False)
# config = dict(type="image,label",
#               image=image_config,
#               label=label_config,
#               manifest_filename='manifest_subset1_augmented.csv',
#               minibatch_size=args.batch_size,
#               subset_fraction=1.0)
# test_set = DataLoader(config, be)
# test_set = TypeCast(test_set, index=0, dtype=np.float32)  # cast image to float

'''
ResNet Model
Taken from https://github.com/NervanaSystems/neon_course/blob/master/06%20Deep%20Residual%20Network.ipynb
'''

# helper functions simplify init params for conv and identity layers
def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm), 
                strides=stride, 
                padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=batch_norm)

def id_params(nfm):
    return dict(fshape=(1, 1, nfm), 
                strides=2, 
                padding=0, 
                activation=None, 
                init=IdentityInit())

# A resnet module
#
#             - Conv - Conv - 
#           /                \
# input   -                   Sum - Relu - output
#           \               /
#            -  Identity - 
#
def module_factory(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**id_params(nfm))]

    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module

# Set depth = 3 for quick results 
# or depth = 9 to reach 6.7% top1 error in 150 epochs
nfms = [2**(stage + 4) for stage in sorted(range(3) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

layers = [Conv(**conv_params(3, 16))]
for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))

layers.append(Pooling('all', op='avg'))

layers.append(Affine(10, init=Kaiming(local=False), 
                     batch_norm=True, activation=Rectlin()))

layers.append(Affine(1, init=Kaiming(local=False), 
                     batch_norm=True, activation=Logistic()))

lunaModel = Model(layers=layers)

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=Schedule([90, 135], 0.1))



# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
    
# configure callbacks
callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
# add a callback that saves the best model state
callbacks.add_save_best_state_callback('LUNA16_resnet_32.prm')


lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

lunaModel.save_params('LUNA16_resnet_32.prm')

# neon_logger.display('Calculating metrics on the test set. This could take a while...')
# neon_logger.display('Misclassification error (test) = {:.2f}%'.format(lunaModel.eval(test_set, metric=Misclassification())[0] * 100))

# neon_logger.display('Precision/recall (test) = {}'.format(lunaModel.eval(test_set, metric=PrecisionRecall(num_classes=2))))


