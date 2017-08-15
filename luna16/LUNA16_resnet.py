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
import itertools as itt
from neon import logger as neon_logger

from neon.optimizers import Adam, Adadelta
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.transforms import Cost
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification, PrecisionRecall
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
parser.add_argument('--depth', type=int, default=18,
                    help='choices 18, 34, 50, 101, 152')

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

patch_size = 64

# Set up the training set to load via aeon
# Augmentating the data via flipping, rotating, changing contrast/brightness
image_config = dict(height=patch_size, width=patch_size, flip_enable=True, channels=3,
                    contrast=(0.9,1.1), brightness=(0.9,1.1), 
                    scale=(0.75,0.75), fixed_aspect_ratio=True)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_all_but_9.csv',
              minibatch_size=args.batch_size,
              macrobatch_size=128,
              cache_directory='cache_dir',
              shuffle_manifest=True,
              shuffle_every_epoch = True)
              #shuffle_every_epoch = True)
train_set = DataLoader(config, be)
train_set = TypeCast(train_set, index=0, dtype=np.float32)  # cast image to float

# Set up the validation set to load via aeon
image_config = dict(height=patch_size, width=patch_size, channels=3)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset9_augmented.csv',
              minibatch_size=args.batch_size)
valid_set = DataLoader(config, be)
valid_set = TypeCast(valid_set, index=0, dtype=np.float32)  # cast image to float


class WeightedBinaryCrossEntropyCost(Cost):

  def __init__(self, scale=1, weight=1.0):
      """
      Args:
          scale (float, optional): Amount by which to scale the backpropagated error (default: 1)
      """
      self.scale = scale
      self.weight = weight

  def __call__(self, y, t):
      """
      Returns the weighted binary cross entropy cost.
      Args:
          y (Tensor or OpTree): Output of previous layer or model
          t (Tensor or OpTree): True targets corresponding to y
      Returns:
          OpTree: Returns the weighted cross entropy cost
      """
      assert y.shape == t.shape, "Weighted CrossEntropy requires network output shape to match targets"
      return self.be.sum((1-self.weight)*self.be.safelog(1 - y) * (t - 1) - self.weight*self.be.safelog(y) * t, axis=0)

  def bprop(self, y, t):
      """
      Returns the derivative of the weighted cross entropy cost.
      Args:
          y (Tensor or OpTree): Output of previous layer or model
          t (Tensor or OpTree): True targets corresponding to y
      Returns:
          OpTree: Returns the (mean) shortcut derivative of the binary entropy
                  cost function ``(y - t) / y.shape[1]``
      """
      return self.scale * self.weight * (y - t)


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
    layers.append(Affine(10, init=Kaiming(local=False), 
                     batch_norm=True, activation=Rectlin()))
    layers.append(Affine(1, init=Kaiming(local=False), activation=Logistic()))
    #return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyBinary())
    return Model(layers=layers), GeneralizedCost(costfunc=CrossEntropyBinary())

lunaModel, cost = create_network(args.depth)

modelFileName = 'LUNA16_resnet.prm'
# If model file exists, then load the it and start from there.
# if (os.path.isfile(modelFileName)):
#   lunaModel = Model(modelFileName)

weight_sched = Schedule([30, 60], 0.1)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=weight_sched)

# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
    
# configure callbacks
callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
# add a callback that saves the best model state
callbacks.add_save_best_state_callback(modelFileName)


lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

lunaModel.save_params(modelFileName)

# neon_logger.display('Calculating metrics on the test set. This could take a while...')
# neon_logger.display('Misclassification error (test) = {:.2f}%'.format(lunaModel.eval(test_set, metric=Misclassification())[0] * 100))

# neon_logger.display('Precision/recall (test) = {}'.format(lunaModel.eval(test_set, metric=PrecisionRecall(num_classes=2))))


