#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2017 Intel Nervana Systems Inc.
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
Auto-Encoder implementation in neon.

We'll use a simple auto-encoder topology to create a model
that removes noise from images.

To do this, we'll load the MNIST digits dataset. We then add noise
to the images. The input of the model is the image with noise. 
The output of the model is the image without noise.

"""
import numpy as np
from neon import logger as neon_logger
from neon.data import ArrayIterator, MNIST
from neon.initializers import Uniform
from neon.layers import Conv, Pooling, GeneralizedCost, Deconv
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, SumSquared, Logistic, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

width = 28   # Image width
height = 28  # Image height
amount_of_noise = 1.0  # Positive number (usually between 0-1)

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# Load dataset
dataset = MNIST(path=args.data_dir)
(X_train, y_train), (X_test, y_test), nclass = dataset.load_data()

y_train = X_train # In an autoencoder the output predicts the input
# Add some random noise to the input images
X_train += np.random.uniform(low=0.0, high=amount_of_noise, size=np.shape(X_train))

y_test = X_test # In an autoencoder the output predicts the input
# Add some random noise to the input images
X_test += np.random.uniform(low=0.0, high=amount_of_noise, size=np.shape(X_test))

# Create iterators for the training and testing sets
train_set = ArrayIterator(X=X_train, y=y_train, lshape=(1, height, width), make_onehot=False)
test_set = ArrayIterator(X=X_test, y=y_test, lshape=(1, height, width), make_onehot=False)

# Initialize the weights and the learning rule
init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.001, momentum_coef=0.9)

# Strided conv autoencoder
bn = False
layers = [Conv((4, 4, 8), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling(2),
          Conv((4, 4, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling(2),
          Deconv(fshape=(4, 4, 8), init=init_uni,
                 activation=Rectlin(), batch_norm=bn),
          Deconv(fshape=(3, 3, 8), init=init_uni,
                 activation=Rectlin(), strides=2, batch_norm=bn),
          Deconv(fshape=(2, 2, 1), init=init_uni, activation=Logistic(), strides=2, padding=1)]

# Define the cost
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(model, **args.callback_args)

# Fit the model
model.fit(train_set, optimizer=opt_gdm, num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)

# Let's predict the test set with the current model
results = model.get_outputs(test_set)

# Plot the predicted images
try:
    import matplotlib.pyplot as plt
    fi = 0

    # Plot a 10x12 set of predictions and originals
    nrows = 10
    ncols = 12   

    preds = np.zeros((height * nrows, width * ncols))
    origs = np.zeros((height * nrows, width * ncols))

    idxs = [(row, col) for row in range(nrows) for col in range(ncols)]

    for row, col in idxs:

        im = results[fi,:].reshape((height, width))
        preds[height * row:height * (row + 1):, width * col:width * (col + 1)] = im

        im = X_test[fi,:].reshape(height, width)
        origs[height * row:height * (row + 1):, width * col:width * (col + 1)] = im

        fi = fi + 1

    plt.subplot(1,2,1)
    plt.imshow(preds, cmap='gray')
    plt.title('Predicted masks')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(origs, cmap='gray')
    plt.title('Original images')
    plt.axis('off')

    plt.savefig('Reconstructed.png')

except ImportError:
    neon_logger.display(
        'matplotlib needs to be manually installed to generate plots\npip install matplotlib')