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
CADIMI on LUNA16 data.

Based on the model from the paper:
FALSE POSITIVE REDUCTION FOR NODULE DETECTION IN CT SCANS USING A CONVOLUTIONAL NEURAL NETWORK: APPLICATION TO THE LUNA16 CHALLENGE
Thomas de Bel, Cas van den Bogaard, Valentin Kotov, Luuk Scholten, Nicole Walasek
https://luna16.grand-challenge.org/serve/public_html/pdfs/CADIMI-TEAM1_160816.pdf/

"""
import itertools as itt
from neon import logger as neon_logger

from neon.optimizers import GradientDescentMomentum
from neon.transforms import Cost, Softmax
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.models import Model
from aeon import DataLoader
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from neon.data.dataloader_transformers import TypeCast
import numpy as np

from neon.initializers import Kaiming, IdentityInit, Constant
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation, Dropout

from neon.data.datasets import Dataset
from neon.util.persist import load_obj
import os

from neon.data import HDF5IteratorOneHot


# parse the command line arguments
parser = NeonArgparser(__doc__)
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

relu = Rectlin()
bn = True
convp1 = dict(init=Kaiming(local=True), batch_norm=bn, activation=relu, padding=1)

layers = [Conv((5, 5, 24), **convp1),
          Pooling(2, op='max'),
          Conv((3, 3, 32), **convp1),
          Pooling(2, op='max'),
          Conv((3, 3, 48), **convp1),
          #Dropout(keep=.6),
          Pooling(2, op='max'),

          Affine(nout=512, init=Kaiming(), activation=relu),
          #Dropout(keep=.4),
          Affine(nout=2, init=Kaiming(), activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

lunaModel = Model(layers=layers)


modelFileName = 'LUNA16_CADIMI_subset{}.prm'.format(SUBSET)

# If model file exists, then load the it and start from there.
# if (os.path.isfile(modelFileName)):
#   lunaModel = Model(modelFileName)

# Nesterov accel- erated gradient descent with a learning rate of 0.01, a decay of 10^-3 and a momentum of 0.9
opt = GradientDescentMomentum(0.01, 0.9, wdecay=0.001, nesterov=True)

# configure callbacks
if args.callback_args['eval_freq'] is None:
    args.callback_args['eval_freq'] = 1
    
# configure callbacks
callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
# add a callback that saves the best model state
callbacks.add_save_best_state_callback(modelFileName)


lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)


