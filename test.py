# as first layer in a sequential model:
# as first layer in a sequential model:
from keras.layers import Convolution2D, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Convolution2D(64, 3, 3,
                        border_mode='same',
                        input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
