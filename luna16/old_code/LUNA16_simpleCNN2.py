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

Command:
python LUNA16_simpleCNN2.py -z 64 -e 200 -i 0 -b gpu

On subset0 train, subset1 validation, this gives:

Misclassification error = 13.95%
Precision/recall = [ 0.82056659  0.77919328]

"""

from neon import logger as neon_logger
from neon.initializers import Gaussian, GlorotUniform
from neon.optimizers import Adam, Adadelta
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost, Affine
from neon.transforms import Rectlin, Softmax, Logistic, CrossEntropyMulti, CrossEntropyBinary, Misclassification, PrecisionRecall
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

# Next line gets rid of the deterministic warning
args.deterministic = None

if (args.rng_seed is None):
  args.rng_seed = 16

print('Batch size = {}'.format(args.batch_size))

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))
be.enable_winograd = 4  # default to winograd 4 for fast autotune

# Set up the training set to load via aeon
# Augmentating the data via flipping, rotating, changing contrast/brightness
image_config = dict(height=64, width=64, flip_enable=True, channels=1,
                    contrast=(0.5,1.0), brightness=(0.5,1.0), 
                    scale=(0.7,1), fixed_aspect_ratio=True)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset0_augmented.csv',
              minibatch_size=args.batch_size,
              shuffle_every_epoch = True)
train_set = DataLoader(config, be)
train_set = TypeCast(train_set, index=0, dtype=np.float32)  # cast image to float
train_set = OneHot(train_set, index=1, nclasses=2)

# Set up the validation set to load via aeon
image_config = dict(height=64, width=64, channels=1)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset1_augmented.csv',
              minibatch_size=args.batch_size)
valid_set = DataLoader(config, be)
valid_set = TypeCast(valid_set, index=0, dtype=np.float32)  # cast image to float
valid_set = OneHot(valid_set, index=1, nclasses=2)

# Set up the testset to load via aeon
image_config = dict(height=64, width=64, channels=1)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename='manifest_subset2_augmented.csv',
              minibatch_size=args.batch_size,
              subset_fraction=1.0)
test_set = DataLoader(config, be)
test_set = TypeCast(test_set, index=0, dtype=np.float32)  # cast image to float
test_set = OneHot(test_set, index=1, nclasses=2)

#init_uni = Gaussian(scale=0.05)
init_uni = GlorotUniform()
#opt_gdm = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)
opt_gdm = Adadelta(decay=0.95, epsilon=1e-6)

relu = Rectlin()
bn = True
conv = dict(init=init_uni, batch_norm=bn, activation=relu)
convp1 = dict(init=init_uni, batch_norm=bn, activation=relu, padding=1)
convp1s2 = dict(init=init_uni, batch_norm=bn,
                activation=relu, padding=1, strides=2)

layers = [Conv((5, 5, 24), **convp1),
          Pooling(2, op='max'),
          Conv((3, 3, 32), **convp1),
          Pooling(2, op='max'),
          Conv((3, 3, 48), **convp1),
          Dropout(keep=.6),
          Pooling(2, op='max'),
          Affine(nout=64, init=init_uni, activation=relu),
          Dropout(keep=.4),
          Affine(nout=2, init=init_uni, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyBinary())

lunaModel = Model(layers=layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    lunaModel.load_params(args.model_file)

# configure callbacks
#callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
callbacks = Callbacks(lunaModel, eval_set=valid_set, metric=Misclassification(), **args.callback_args)

if args.deconv:
    callbacks.add_deconv_callback(train_set, valid_set)

lunaModel.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs,
        cost=cost, callbacks=callbacks)

lunaModel.save_params('LUNA16_simpleCNN2_model.prm')

neon_logger.display('Finished training. Calculating error on the validation set...')
neon_logger.display('Misclassification error (validation) = {:.2f}%'.format(lunaModel.eval(valid_set, metric=Misclassification())[0] * 100))

neon_logger.display('Precision/recall (validation) = {}'.format(lunaModel.eval(valid_set, metric=PrecisionRecall(num_classes=2))))

neon_logger.display('Calculating metrics on the test set. This could take a while...')
neon_logger.display('Misclassification error (test) = {:.2f}%'.format(lunaModel.eval(test_set, metric=Misclassification())[0] * 100))

neon_logger.display('Precision/recall (test) = {}'.format(lunaModel.eval(test_set, metric=PrecisionRecall(num_classes=2))))
