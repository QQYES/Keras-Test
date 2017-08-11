from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input


def identity_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter
    out = Convolution2D(k1, 1, 1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2, kernel_size, kernel_size, border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, 1, 1)(out)
    out = BatchNormalization()(out)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter

    out = Convolution2D(k1, 1, 1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(k2, kernel_size, kernel_size)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, 1, 1)(out)
    out = BatchNormalization()(out)

    x = Convolution2D(k3, 1, 1)(x)
    x = BatchNormalization()(x)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)
    return out


inp = Input(shape=(3, 224, 224))
out = ZeroPadding2D((3, 3))(inp)
out = Convolution2D(64, 7, 7, subsample=(2, 2))(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
out = MaxPooling2D((3, 3), strides=(2, 2))(out)

out = conv_block(out, [64, 64, 256])
out = identity_block(out, [64, 64, 256])
out = identity_block(out, [64, 64, 256])

out = conv_block(out, [128, 128, 512])
out = identity_block(out, [128, 128, 512])
out = identity_block(out, [128, 128, 512])
out = identity_block(out, [128, 128, 512])

out = conv_block(out, [256, 256, 1024])
out = identity_block(out, [256, 256, 1024])
out = identity_block(out, [256, 256, 1024])
out = identity_block(out, [256, 256, 1024])
out = identity_block(out, [256, 256, 1024])
out = identity_block(out, [256, 256, 1024])

out = conv_block(out, [512, 512, 2048])
out = identity_block(out, [512, 512, 2048])
out = identity_block(out, [512, 512, 2048])

out = AveragePooling2D((7, 7))(out)
out = Dense(1000, activation='softmax')(out)

model = Model(inp, out)

import keras.backend as K
from keras.layers import Activation
import numpy as np

x = K.placeholder(shape=(3,))
y = Activation('sigmoid')(x)
f = K.function([x], [y])
out = f([np.array([1, 2, 3])])
