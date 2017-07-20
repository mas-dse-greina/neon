#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
AllCNN style convnet on LUNA16 data.

"""

from neon import logger as neon_logger
from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from aeon import DataLoader
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

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

# Set up the training set to load via aeon
# Augmentating the data via flipping, rotating, changing contrast/brightness
image_config = dict(height=64, width=64, flip_enable=True, 
                    contrast=(0.5,1.0), brightness=(0.5,1.0), 
                    fixed_aspect_ratio = True, angle=(0,45))
label_config = dict(binary=True)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset0.txt',
              minibatch_size=128)
train_set = DataLoader(config, backend)

# Set up the test set to load via aeon
image_config = dict(height=64, width=64)
label_config = dict(binary=True)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset1.txt',
              minibatch_size=128)
val_set= DataLoader(config, backend)


init_uni = Gaussian(scale=0.05)
opt_gdm = GradientDescentMomentum(learning_rate=float(args.learning_rate), momentum_coef=0.9,
                                  wdecay=float(args.weight_decay),
                                  schedule=Schedule(step_config=[200, 250, 300], change=0.1))

relu = Rectlin()
conv = dict(init=init_uni, batch_norm=False, activation=relu)
convp1 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1)
convp1s2 = dict(init=init_uni, batch_norm=False,
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
          Pooling(2, op="avg"),
          Activation(Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    mlp.load_params(args.model_file)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

if args.deconv:
    callbacks.add_deconv_callback(train_set, valid_set)

mlp.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs,
        cost=cost, callbacks=callbacks)
neon_logger.display('Misclassification error = %.1f%%' %
                    (mlp.eval(valid_set, metric=Misclassification()) * 100))