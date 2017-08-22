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

python LUNA16_inferenceTesting_ResNet_HDF.py -b gpu -z 512 --subset 8

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
from neon.data import HDF5Iterator
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

# Set up the testset to load via aeon
test_set = HDF5Iterator(testFileName)

lunaModel = Model('LUNA16_resnetHDF_subset{}.prm'.format(subset))

def round(arr, threshold=0.5):
   '''
       Round to an arbitrary threshold.
       Above threshold goes to 1. Below goes 0.
   '''
   out = np.zeros(np.shape(arr))
   out[np.where(arr > threshold)[0]] = 1

   out[np.where(arr <= threshold)[0]] = 0

   return out

prob, target = lunaModel.get_outputs(test_set, return_targets=True) 
prob = prob.T[0]
target = target.T[0]
np.set_printoptions(precision=3, suppress=True)
#print(' ')
#print(prob)
#print(' ')

threshold = 0.5
pred = round(prob, threshold=threshold).astype(int)
# print(pred),
# print('predictions')
# print(target),
# print('targets')

# print('Predict 1 count = {}'.format(len(np.where(pred == 1)[0])))
# print('True 1 count= {}'.format(len(np.where(target == 1)[0])))

# print('Predict 0 count = {}'.format(len(np.where(pred == 0)[0])))
# print('True 0 count= {}'.format(len(np.where(target == 0)[0])))

# target_zero_idx = np.where(target == 0)[0]
# target_one_idx = np.where(target == 1)[0]

# false_positive = len(np.where(pred[target_zero_idx] == 1)[0])
# false_negative = len(np.where(pred[target_one_idx] == 0)[0])

# print('False positive count = {}'.format(false_positive))
# print('False negative count = {}'.format(false_negative))

if (True):

  # print('All equal = {}'.format(np.array_equal(pred, target)))

  # print('Incorrect prediction probabilities = {}'.format(prob[np.where(pred != target)[0]]))
  # print('Indices = {}'.format(np.where(pred != target)[0]))
  
  from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
  from sklearn.metrics import precision_recall_curve, log_loss

  precision, recall, thresholds = precision_recall_curve(target, prob)

  print('Average precision = {}'.format(average_precision_score(target, prob, average='weighted')))

  print('Log loss = {}'.format(log_loss(target, prob)))

  print('For threshold of {}:'.format(threshold))
  print(classification_report(target, pred, target_names=['Class 0', 'Class 1']))

  print('Area under the curve = {}'.format(roc_auc_score(target, prob)))

  # neon_logger.display('Calculating metrics on the test set. This could take a while...')

  # misclassification = lunaModel.eval(test_set, metric=Misclassification())
  # neon_logger.display('Misclassification error (test) = {}'.format(misclassification))

  # precision, recall = lunaModel.eval(test_set, metric=PrecisionRecall(num_classes=2))
  # neon_logger.display('Precision = {}, Recall = {}'.format(precision, recall))

  import pandas as pd

  # Save the predictions file
  df = h5py.File(testFileName)
  out_df = pd.DataFrame(df['position'][()], columns=['filename','x','y','z'])
  out_df['prob'] = prob
  out_df.to_csv('predictions{}.csv'.format(subset), header=None, index=None)


