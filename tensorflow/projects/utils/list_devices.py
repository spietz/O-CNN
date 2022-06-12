import os

# TODO: fix incompatible numpy.. supress warning, before loading tf
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
