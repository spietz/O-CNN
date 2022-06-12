import os
import sys

# TODO: fix incompatible numpy.. supress warning, before loading tf
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

sys.path.append("../../../tensorflow")
sys.path.append("../../../tensorflow/script")

from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_factory import cls_network


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# define regression loss function

from ocnn import l2_regularizer

def loss_functions_regress(signal, signal_gt, weight_decay, var_name, label_smoothing=0.0):
  with tf.name_scope('loss_regress'):
    loss = tf.compat.v1.losses.mean_squared_error(signal_gt, signal)
    print(loss)
    accu = tf.sqrt(loss)
    regularizer = l2_regularizer(var_name, weight_decay)
  return [loss, accu, regularizer]


# define data set

from dataset import NormalizePoints, PointDataset, TransformPoints, Points2Octree

class ParseExample:  # points object with with vector
  def __init__(self, x_alias='data', y_alias='label', **kwargs):
    self.x_alias = x_alias
    self.y_alias = y_alias
    self.features = { x_alias : tf.FixedLenFeature([], tf.string),
                      y_alias : tf.FixedLenFeature([36], tf.float32) }

  def __call__(self, record):
    parsed = tf.parse_single_example(record, self.features)
    return parsed[self.x_alias], parsed[self.y_alias]

class DatasetFactory:
  def __init__(self, flags, normalize_points=NormalizePoints,
               point_dataset=PointDataset, transform_points=TransformPoints):
    self.flags = flags
    self.dataset = point_dataset(ParseExample(**flags), normalize_points(), 
      transform_points(**flags), Points2Octree(**flags))

  def __call__(self, return_iter=False):
    return self.dataset(
        record_names=self.flags.location, batch_size=self.flags.batch_size,
        shuffle_size=self.flags.shuffle, return_iter=return_iter,
        take=self.flags.take, return_pts=self.flags.return_pts)     

# configs
FLAGS = parse_args()

#define the graph
def compute_graph(dataset='train', training=True, reuse=False):
  flags_data = FLAGS.DATA.train if dataset=='train' else FLAGS.DATA.test
  octree, label = DatasetFactory(flags_data)()

  logit = cls_network(octree, FLAGS.MODEL, training, reuse)  # reuse cls network, switch loss function
  losses = loss_functions_regress(logit, label,
                                  FLAGS.LOSS.weight_decay, 'ocnn', FLAGS.LOSS.label_smoothing)
  losses.append(losses[0] + losses[2]) # total loss
  names  = ['loss', 'accu', 'regularizer', 'total_loss']
  return losses, names

# run the experiments
if __name__ == '__main__':
  solver = TFSolver(FLAGS, compute_graph)
  solver.run()

# flags_data = FLAGS.DATA.train
# octree, label = DatasetFactory(flags_data)()
