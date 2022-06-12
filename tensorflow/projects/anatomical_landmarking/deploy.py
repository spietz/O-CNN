import os
import sys
import numpy as np
import json

# TODO: fix incompatible numpy.. supress warning, before loading tf
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.append("../../../tensorflow")
sys.path.append("../../../tensorflow/script")

from config import parse_args
from network_factory import cls_network
from ocnn import octree_batch

sys.path.append("..")
import utils

import vtk

# run using
# deploy.py --config configs/deploy_lm_5_2.yaml DEPLOY.input test.vtk

# ordered list of names of landmarks

lm_keys = [
    'center.lips.upper.outer',
    'center.nose.tip',
    'left.lips.corner',
    'left.ear.helix.attachement',
    'left.ear.tragus.tip',
    'left.eye.corner_inner',
    'left.eye.corner_outer',
    'right.lips.corner',
    'right.ear.helix.attachement',
    'right.ear.tragus.tip',
    'right.eye.corner_inner',
    'right.eye.corner_outer'
]


# configs:

FLAGS = parse_args()


# load data ####################################################################

mesh_file_str = FLAGS.DEPLOY.input

assert(mesh_file_str)

# mesh to points
extractor = utils.PointsExtractor()
points, normals = extractor.extract(mesh_file_str, FLAGS.MODEL.depth)

# look for json file with true landmarks in the same directory
lm_file_str = mesh_file_str.replace("vtk", "json")
lm_file_exists = os.path.exists(lm_file_str)

if lm_file_exists:

  # load json
  with open(lm_file_str) as f:
      data = json.load(f)
  landmarks = {}
  for entry in data:
      landmarks[entry['id']]=entry['coordinates']

  # loop landmarks in the predefined order to populate np array
  lm_true = np.zeros((len(lm_keys),3))
  for i, key in enumerate(lm_keys):
      if key not in landmarks:
          print("%s does not have landmark %s" % (lm_file_str, key))
          continue
      lm_true[i, :] = landmarks[key]


# setup input pipeline and network #############################################

from libs import points_new, bounding_sphere, normalize_points, points2octree

""" x = tf.placeholder(tf.)  # read octree (tf.string) """

x_pts = tf.placeholder(tf.float32, shape=[None])
x_nms = tf.placeholder(tf.float32, shape=[None])

x = points_new(points, normals, [], [])

radius, center = bounding_sphere(x)  # get bounds for rescaling output

x = normalize_points(x, radius, center)

depth      = FLAGS.MODEL.depth          # The octree depth
full_depth = 2          # The full depth
node_dis   = False      # Save the node displacement
node_feat  = False      # Calculate the node feature
split_label= False      # Save the split label
adaptive   = False      # Build the adaptive octree
adp_depth  = 4
th_normal  = 0.1
save_pts   = False

x = points2octree(x, depth, full_depth,
                  node_dis, node_feat, 
                  split_label, adaptive, 
                  adp_depth, th_normal,
                  save_pts)

# octree_batch tf.string -> tf.int8
y = cls_network(octree_batch(x), FLAGS.MODEL, training=False, reuse=False)


# load trained coefficients and process input ##################################

assert(FLAGS.SOLVER.ckpt)

tf_saver = tf.train.Saver(max_to_keep=10)

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  
  sess.run(tf.global_variables_initializer())
  
  tf_saver.restore(sess, FLAGS.SOLVER.ckpt)

  rad, cen, y_predict = sess.run([radius, center, y], feed_dict={x_pts:points.flatten().tolist(), x_nms:normals.flatten().tolist()})



# rescale
lm_predict = np.reshape(y_predict, (-1, 3))*rad + cen

print("done processing: %s" % mesh_file_str)


# dump as vtk files ############################################################

for i, k in enumerate(lm_keys):
  #print("%-50s" % k, lm_predict[i,:])
  utils.writePoints(
    "predict.vtk",lm_predict)

if lm_file_exists:
  for i, k in enumerate(lm_keys):
    # print("%-50s" % k, lm_true[i,:])
    utils.writePoints("true.vtk", lm_true)



# vizualisation ################################################################

from utils import vtk_utils
import vtk

colors = vtk.vtkNamedColors()

# reload polydata
mesh_pd = vtk_utils.getPolydata(mesh_file_str)
predict_pd = vtk_utils.getPolydata("predict_%d_%d.vtk" % (FLAGS.MODEL.depth, full_depth))
true_pd = vtk_utils.getPolydata("true.vtk")

# plot mesh
meshMapper = vtk.vtkPolyDataMapper()
meshMapper.SetInputData(mesh_pd)

meshActor = vtk.vtkActor()
meshActor.SetMapper(meshMapper)
meshActor.GetProperty().SetOpacity(0.8)

# plot points as spheres
sphereSource = vtk.vtkSphereSource()
sphereSource.SetRadius(2.0)

# plot predicted points
predictGlyphs = vtk.vtkGlyph3D()
predictGlyphs.SetSourceConnection(sphereSource.GetOutputPort())
predictGlyphs.SetInputData(predict_pd)

predictMapper = vtk.vtkPolyDataMapper()
predictMapper.SetInputConnection(predictGlyphs.GetOutputPort())
predictMapper.Update()

predictActor = vtk.vtkActor()
predictActor.SetMapper(predictMapper)
predictActor.GetProperty().SetColor(colors.GetColor3d("RED"))

# plot true points
trueGlyphs = vtk.vtkGlyph3D()
trueGlyphs.SetSourceConnection(sphereSource.GetOutputPort())
trueGlyphs.SetInputData(true_pd)

trueMapper = vtk.vtkPolyDataMapper()
trueMapper.SetInputConnection(trueGlyphs.GetOutputPort())
trueMapper.Update()

trueActor = vtk.vtkActor()
trueActor.SetMapper(trueMapper)
trueActor.GetProperty().SetColor(colors.GetColor3d("LIME"))

# Create the graphics structure. The renderer renders into the render
# window. The render window interactor captures mouse events and will
# perform appropriate camera or actor manipulation depending on the
# nature of the events.
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set size
ren.AddActor(meshActor)
ren.AddActor(predictActor)
ren.AddActor(trueActor)
renWin.SetSize(800, 800)
renWin.SetWindowName('Anatomical landmark prediction')

# Fix opacity
ren.SetUseDepthPeeling(1)
ren.SetOcclusionRatio(0.1)
ren.SetMaximumNumberOfPeels(100)
renWin.SetMultiSamples(0)
renWin.SetAlphaBitPlanes(1)

# This allows the interactor to initalize itself. It has to be
# called before an event loop.
iren.Initialize()

# We'll zoom in a little by accessing the camera and invoking a "Zoom"
# method on it.
ren.ResetCamera()
ren.GetActiveCamera().Zoom(1.5)
renWin.Render()

# Start the event loop.
iren.Start()