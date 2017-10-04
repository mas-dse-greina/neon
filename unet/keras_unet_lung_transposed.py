#!/usr/bin/env python

'''
BEGIN - Limit Tensoflow to only use specific GPU
'''
import os

gpu_num = 2

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress Tensforflow debug messages

import tensorflow as tf

'''
END - Limit Tensoflow to only use specific GPU
'''

import numpy as np # linear algebra
import cv2
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import to_categorical

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # Use only 80% of available gpu memory
sess = tf.Session(config=config)
K.set_session(sess)

IMAGE_LIB = 'finding-lungs-in-ct-data/2d_images/'
MASK_LIB = 'finding-lungs-in-ct-data/2d_masks/'

IMG_HEIGHT, IMG_WIDTH = 256, 256
SEED=16


def dice_coef(y_true, y_pred, smooth = 1. ):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return coef

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def img_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch

'''
Get all of the images in the data directory
'''
all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.tif']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, cv2.IMREAD_UNCHANGED).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im

x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.3)

print('X train shape = {}'.format(x_train.shape))
print('Y train shape = {}'.format(y_train.shape))


def create_unet_model(USE_ORIGINAL = True):

    '''
    This is based on the original paper. 
    Olaf Ronneberger, Philipp Fischer, and Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/pdf/1505.04597.pdf

    If USE_ORIGINAL is true, then follow the original UpPooling from the paper. If not, the replace
    UpPooling with a Transposed Convolution.
    '''

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))

    # "Contracting path" (down the left side of the U)
    # Each level doubles the number of feature maps but halves the map size
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='contract1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='contract2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='contract3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='contract4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same', name='contract5')(conv5)

    # "Expansive path" (up the right side of the U)
    # With each up layer, we concatenate
    # the features maps from the downsampling encoder part so that our
    # classifier has features from multiple receptive field scales.
    # There's a choice in how we should do the upward sampling.
    # In the paper they perform an UpPooling to increase the feature map height and width.
    # We could also replace the UpPooling with a Transposed Convolution to do the same thing.
    

    if USE_ORIGINAL:
        conv6 = UpSampling2D(size=(2,2), name='expand6_up')(conv5)
    else:
        conv6 = Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), padding='same', name='expand6_trans')(conv5)
    
    concat6 = concatenate([conv6, conv4], axis=3)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(concat6)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(conv6)

    if USE_ORIGINAL:
        conv7 = UpSampling2D(size=(2,2), name='expand7_up')(conv6)
    else:
        conv7 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', name='expand7_trans')(conv6)
    
    concat7 = concatenate([conv7, conv3], axis=3)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(concat7)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv7)

    if USE_ORIGINAL:
        conv8 = UpSampling2D(size=(2,2), name='expand8_up')(conv7)
    else:
        conv8 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', name='expand8_trans')(conv7)
    
    concat8 = concatenate([conv8, conv2], axis=3)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(concat8)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv8)

    if USE_ORIGINAL:
        conv9 = UpSampling2D(size=(2,2), name='expand9_up')(conv8)
    else:
        conv9 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', name='expand9_trans')(conv8)
        

    concat9 = concatenate([conv9, conv1], axis=3)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concat9)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv9)

    output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', name='Output')(conv9)
               
    model = Model(inputs=[inputs], outputs=[output_layer])

    # The paper uses binary_crossentropy for the loss function. However, dice_coefficient has been
    # recommended as a better loss function for segmentation models.
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


model = create_unet_model(USE_ORIGINAL=True)

model.summary()

save_weights = ModelCheckpoint('lung.h5', monitor='val_dice_coef', 
                                save_best_only=True, save_weights_only=True)

batch_size = 8

hist = model.fit_generator(img_generator(x_train, y_train, batch_size),
                           steps_per_epoch = 200, #x_train.shape[0]//batch_size,
                           shuffle=True,
                           validation_data = (x_val, y_val),
                           epochs=10, verbose=1,
                           callbacks = [save_weights])



import matplotlib.pyplot as plt

model.load_weights('lung.h5')

y_hat = model.predict(x_val)

np.save('y_hat', y_hat)
np.save('y_val', y_val)
np.save('x_val', x_val)

fig, ax = plt.subplots(1,3,figsize=(12,6))
ax[0].imshow(x_val[0,:,:,0], cmap='gray')
ax[1].imshow(y_val[0,:,:,0])
ax[2].imshow(y_hat[0,:,:,0])

plt.savefig('lung_segmented1.png', dpi=600)

imgNum = 10

fig, ax = plt.subplots(1,3,figsize=(12,6))
ax[0].imshow(x_val[imgNum,:,:,0], cmap='gray')
ax[1].imshow(y_val[imgNum,:,:,0])
ax[2].imshow(y_hat[imgNum,:,:,0])

plt.savefig('lung_segmented{}.png'.format(imgNum), dpi=600)

print('FINISHED Keras UNet.')


