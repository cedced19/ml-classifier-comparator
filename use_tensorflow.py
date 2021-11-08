import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))