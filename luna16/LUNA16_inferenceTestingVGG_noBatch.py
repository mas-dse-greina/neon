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
Load and test a pre-trained model against the entire data subset.

python LUNA16_inferenceTesting.py -b gpu -i 0 -z 16

"""

from neon import logger as neon_logger
from neon.transforms import Misclassification, PrecisionRecall
from neon.models import Model
from aeon import DataLoader
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.backends import gen_backend
from neon.data.dataloader_transformers import TypeCast, OneHot
import numpy as np

# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

testFileName = 'manifest_subset7_SMALL.csv'

# hyperparameters
num_epochs = args.epochs

# Next line gets rid of the deterministic warning
args.deterministic = None

if (args.rng_seed is None):
  args.rng_seed = 16

print('Batch size = {}'.format(args.batch_size))

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# Set up the testset to load via aeon
image_config = dict(height=64, width=64, channels=1)
label_config = dict(binary=False)
config = dict(type="image,label",
              image=image_config,
              label=label_config,
              manifest_filename=testFileName,
              minibatch_size=args.batch_size,
              subset_fraction=1)
test_set = DataLoader(config, be)
test_set = TypeCast(test_set, index=0, dtype=np.float32)  # cast image to float
test_set = OneHot(test_set, index=1, nclasses=2)

lunaModel = Model('LUNA16_VGG_model_no_batch.prm')

pred, target = lunaModel.get_outputs(test_set, return_targets=True) # Reshape to a single prediction vector
pred = pred.T
target = target.T

np.set_printoptions(precision=3, suppress=True)
print(' ')
print(pred)
print(target)

