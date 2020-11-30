import numpy as np 
import tensorflow as tf
import tensorflow.keras as tfk

import u_net3

INPUT_DIM = [132, 132, 116]
OUTPUT_DIM = [44, 44, 28]
NO_CHANNELS = 3
NO_CLASSES = 3
NO_FILTERS = 32

unet_model = u_net3.UNet3D(in_channels=NO_CHANNELS, out_classes=NO_CLASSES, img_shape = [INPUT_DIM[0], INPUT_DIM[1], INPUT_DIM[2], NO_CHANNELS], no_filters=NO_FILTERS)
unet_model.build(input_shape=(1, INPUT_DIM[0], INPUT_DIM[1], INPUT_DIM[2], NO_CHANNELS))
unet_model.summary()

print(1)