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
CNN layers into a simple MLP with softmax output on LUNA16 data.
Just tests that aeon dataloader works and can be fed to simple model in neon.

"""

from neon import logger as neon_logger
from neon.initializers import Gaussian
from neon.optimizers import Adam
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost, Affine
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification, PrecisionRecall
from neon.models import Model
from aeon import DataLoader
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from neon.data.dataloader_transformers import BGRMeanSubtract, TypeCast, OneHot
import numpy as np

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument("--learning_rate", default=0.05,
                    help="initial learning rate")
parser.add_argument("--weight_decay", default=0.001, help="weight decay")
parser.add_argument('--deconv', action='store_true',
                    help='save visualization data from deconvolution')
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs

if (args.deterministic is None):
  args.deterministic = None

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.enable_winograd = 4  # default to winograd 4 for fast autotune

# Set up the training set to load via aeon
# Augmentating the data via flipping, rotating, changing contrast/brightness
image_config = dict(height=64, width=64, flip_enable=True, 
                    contrast=(0.5,1.0), brightness=(0.5,1.0), 
                    fixed_aspect_ratio = True, angle=(0,45))
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_allbut0.csv', 
              subset_fraction=0.01,
              minibatch_size=128)
train_set = DataLoader(config, be)
train_set = TypeCast(train_set, index=0, dtype=np.float32)  # cast image to float
train_set = OneHot(train_set, index=1, nclasses=2)

# Set up the test set to load via aeon
image_config = dict(height=64, width=64)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset0.txt',
              minibatch_size=128)
valid_set = DataLoader(config, be)
valid_set = TypeCast(valid_set, index=0, dtype=np.float32)  # cast image to float
valid_set = OneHot(valid_set, index=1, nclasses=2)

init_uni = Gaussian(scale=0.05)
opt_gdm = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)

relu = Rectlin()
conv = dict(init=init_uni, batch_norm=True, activation=relu)
convp1 = dict(init=init_uni, batch_norm=True, activation=relu, padding=1)
convp1s2 = dict(init=init_uni, batch_norm=True,
                activation=relu, padding=1, strides=2)

layers = [Dropout(keep=.8),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((1, 1, 192), **conv),
          Conv((1, 1, 16), **conv),
          Affine(nout=512, init=init_uni, activation=relu),
          Affine(nout=128, init=init_uni, activation=relu),
          Affine(nout=64, init=init_uni, activation=relu),
          Affine(nout=2, init=init_uni, activation=Softmax())]


cost = GeneralizedCost(costfunc=CrossEntropyMulti())

lunaModel = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    lunaModel.load_params(args.model_file)

# configure callbacks
#callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
callbacks = Callbacks(lunaModel, eval_set=valid_set, metric=PrecisionRecall(num_classes=2), **args.callback_args)

if args.deconv:
    callbacks.add_deconv_callback(train_set, valid_set)

lunaModel.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs,
        cost=cost, callbacks=callbacks)

lunamodel.save_params('LUNA16_simpleCNN_model.prm')

neon_logger.display('Finished training. Calculating error on the validation set...')
neon_logger.display('Misclassification error = {:.1f}'.format(lunaModel.eval(valid_set, metric=Misclassification()) * 100))

neon_logger.display('Precision/recall = {}'.format(lunaModel.eval(valid_set, metric=PrecisionRecall(num_classes=2))))
