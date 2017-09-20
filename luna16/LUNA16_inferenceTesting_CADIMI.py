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


"""

from neon import logger as neon_logger
from neon.models import Model
from aeon import DataLoader
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.transforms import Misclassification, PrecisionRecall
from neon.backends import gen_backend
from neon.data.dataloader_transformers import TypeCast, OneHot
import numpy as np
import pandas as pd
from neon.data import HDF5IteratorOneHot, HDF5Iterator
import h5py

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--subset', type=int, default=9)
args = parser.parse_args()

subset = args.subset
#testFileName = '/mnt/data/medical/luna16/luna16_roi_subset{}_augmented.h5'.format(subset)
testFileName = '/mnt/data/medical/luna16/luna16_roi_subset{}_ALL.h5'.format(subset)

print('Using test file: {}'.format(testFileName))

# Next line gets rid of the deterministic warning
args.deterministic = None

if (args.rng_seed is None):
  args.rng_seed = 16

print('Batch size = {}'.format(args.batch_size))

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

test_set = HDF5Iterator(testFileName)

model_filename= 'LUNA16_CADIMI_subset{}.prm'.format(subset)

print('Using model: {}'.format(model_filename))

lunaModel = Model(model_filename)

prob, target = lunaModel.get_outputs(test_set, return_targets=True) 

np.set_printoptions(precision=3, suppress=True)


from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, log_loss, confusion_matrix

# precision, recall, thresholds = precision_recall_curve(target, prob)

print('Average precision = {}'.format(average_precision_score(target, prob[:,1], average='weighted')))


print(classification_report(target, np.argmax(prob, axis=1), target_names=['Class 0', 'Class 1']))

print('Area under the curve = {}'.format(roc_auc_score(target, prob[:,1])))

pred = np.argmax(prob, axis=1)

[[tn, fp], [fn, tp]] = confusion_matrix(target, pred, labels=[0, 1])
print('True positives = {}, True negatives = {}, False positives = {}, False negatives = {}'.format(tp, tn, fp, fn))

import pandas as pd

# Save the predictions file
df = h5py.File(testFileName)
out_df = pd.DataFrame(df['position'][()], columns=['seriesuid','x','y','z'])
out_df['prob'] = prob[:,1]
#out_df['pred'] = pred
#out_df['target'] = target
out_df.to_csv('predictions{}.csv'.format(subset), index=None, header=None)


