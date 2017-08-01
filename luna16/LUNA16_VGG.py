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
VGG16 on LUNA16 data.

Command:
python LUNA16_VGG.py -z 64 -e 200 -b gpu -i 0



"""

from neon import logger as neon_logger
from neon.initializers import Gaussian, GlorotUniform, Xavier, Constant
from neon.optimizers import Adam, Adadelta
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost, Affine
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Logistic, CrossEntropyBinary, Misclassification, PrecisionRecall
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
              minibatch_size=args.batch_size)
train_set = DataLoader(config, be)
train_set = TypeCast(train_set, index=0, dtype=np.float32)  # cast image to float
#train_set = OneHot(train_set, index=1, nclasses=2)

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
#valid_set = OneHot(valid_set, index=1, nclasses=2)

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
#test_set = OneHot(test_set, index=1, nclasses=2)

#init_uni = Gaussian(scale=0.05)
init_uni = GlorotUniform()
#opt_gdm = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)
opt_gdm = Adadelta(decay=0.95, epsilon=1e-6)

relu = Rectlin()
conv_params = {'strides': 1,
               'padding': 1,
               'init': Xavier(local=True),
               'bias': Constant(0),
               'activation': relu,
               'batch_norm': False}

# Set up the model layers
vgg_layers = []

# set up 3x3 conv stacks with different number of filters
vgg_layers.append(Conv((3, 3, 64), **conv_params))
vgg_layers.append(Conv((3, 3, 64), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 128), **conv_params))
vgg_layers.append(Conv((3, 3, 128), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Conv((3, 3, 256), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Conv((3, 3, 512), **conv_params))
vgg_layers.append(Pooling(2, strides=2))
vgg_layers.append(Affine(nout=4096, init=GlorotUniform(), bias=Constant(0), activation=relu))
vgg_layers.append(Dropout(keep=0.5))
vgg_layers.append(Affine(nout=4096, init=GlorotUniform(), bias=Constant(0), activation=relu))
vgg_layers.append(Dropout(keep=0.5))

vgg_layers.append(Affine(nout=1, init=GlorotUniform(), bias=Constant(0), activation=Logistic(),
                  name="class_layer"))

# define different optimizers for the class_layer and the rest of the network
# we use a momentum coefficient of 0.9 and weight decay of 0.0005.
opt_vgg = GradientDescentMomentum(0.001, 0.9, wdecay=0.0005)
opt_class_layer = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005)

# also define optimizers for the bias layers, which have a different learning rate
# and not weight decay.
opt_bias = GradientDescentMomentum(0.002, 0.9)
opt_bias_class = GradientDescentMomentum(0.02, 0.9)

# set up the mapping of layers to optimizers
opt = MultiOptimizer({'default': opt_vgg, 'Bias': opt_bias,
     'class_layer': opt_class_layer, 'class_layer_bias': opt_bias_class})

# use cross-entropy cost to train the network
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

lunaModel = Model(layers=vgg_layers)

if args.model_file:
    import os
    assert os.path.exists(args.model_file), '%s not found' % args.model_file
    lunaModel.load_params(args.model_file)

# configure callbacks
#callbacks = Callbacks(lunaModel, eval_set=valid_set, **args.callback_args)
callbacks = Callbacks(lunaModel, eval_set=valid_set, metric=Misclassification(), **args.callback_args)

if args.deconv:
    callbacks.add_deconv_callback(train_set, valid_set)

lunaModel.fit(train_set, optimizer=opt, num_epochs=num_epochs,
        cost=cost, callbacks=callbacks)

lunaModel.save_params('LUNA16_VGG_model.prm')

neon_logger.display('Finished training. Calculating error on the validation set...')
neon_logger.display('Misclassification error (validation) = {:.2f}%'.format(lunaModel.eval(valid_set, metric=Misclassification())[0] * 100))

neon_logger.display('Precision/recall (validation) = {}'.format(lunaModel.eval(valid_set, metric=PrecisionRecall(num_classes=2))))

neon_logger.display('Calculating metrics on the test set. This could take a while...')
neon_logger.display('Misclassification error (test) = {:.2f}%'.format(lunaModel.eval(test_set, metric=Misclassification())[0] * 100))

neon_logger.display('Precision/recall (test) = {}'.format(lunaModel.eval(test_set, metric=PrecisionRecall(num_classes=2))))

