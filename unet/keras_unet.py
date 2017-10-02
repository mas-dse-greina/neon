import os
import tensorflow as tf
from keras import backend as K

def get_session(gpu_fraction=0.4):
    '''
    Don't use up all the GPU memory 
    '''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(device_count = {'GPU': 3},
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

K.set_session(get_session())


from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256
color_channels = 3
num_classes = 2  # 2 output classes in the segmentation map

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def create_unet(input_size=(img_rows, img_cols, color_channels), num_classes=2):

    inputs = Input(input_size)

    # Encoder part (down the left side of the U)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder part (up the right side of the U)
    # Here we just need to do a transposed convolution to up-sample the feature maps
    # back to the original size of the image. With each up layer, we concatenate
    # the features maps from the downsampling encoder part so that our
    # classifier has features from multiple receptive field scales.
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = create_unet((img_rows, img_cols, color_channels), num_classes)
print(model.summary())

# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90.,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

val_datagen = ImageDataGenerator()

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

# Train data generator
image_generator_train = image_datagen.flow_from_directory(
    '/mnt/data/medical/melanoma_isic2017/data/training',
    class_mode=None,
    target_size=(img_rows,img_cols), # Resize the images to these dimensions
    seed=seed)

# Train data mask
mask_generator_train = mask_datagen.flow_from_directory(
    '/mnt/data/medical/melanoma_isic2017/masks/training',
    class_mode=None,
    target_size=(img_rows,img_cols), # Resize the images to these dimensions
    seed=seed)

# Validation data
image_generator_val = val_datagen.flow_from_directory(
    '/mnt/data/medical/melanoma_isic2017/data/validation',
    class_mode=None,
    target_size=(img_rows,img_cols), # Resize the images to these dimensions
    seed=seed)

# Validation mask
mask_generator_val = mask_datagen.flow_from_directory(
    '/mnt/data/medical/melanoma_isic2017/masks/validation',
    class_mode=None,
    target_size=(img_rows,img_cols), # Resize the images to these dimensions
    seed=seed)


# combine generators into one which yields image and masks
train_generator = zip(image_generator_train, mask_generator_train)

validation_generator = zip(image_generator_val, mask_generator_val)


model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50, 
    validation_data=validation_generator,
    validation_steps=800)

