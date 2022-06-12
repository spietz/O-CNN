import numpy as np
import vtk
from vtk.util import numpy_support

# get the polydata object
def getPolydata(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# get the polydata object
def writePolydata(filename, polydata):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

# get points as array (npoints, ndim)
def getPoints(polydata):
    vtkDataArray = polydata.GetPoints().GetData()
    return numpy_support.vtk_to_numpy(vtkDataArray)

# get triangle cells connectivity as array (ncells, 3)
def getCellIds(polydata):
    cells = polydata.GetPolys()
    ids = []
    idList = vtk.vtkIdList()
    cells.InitTraversal()
    while cells.GetNextCell(idList):
        cellIdList = []
        for i in range(0, idList.GetNumberOfIds()):
            pId = idList.GetId(i)
            cellIdList.append(pId)
        ids.append(cellIdList)
    return np.array(ids)

# get point normals as array (npoints, ndim) (must be from vtk.vtkPolyDataNormals())
def getNormals(polydata):
    pts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    nms = numpy_support.vtk_to_numpy(polydata.GetPointData().GetNormals())
    return pts, nms  # also points because seems to be different

# write numpy array (N,3) of points to file
def writePoints(filename, ndarray):
    assert(ndarray.shape[1]==3)

    # Create the geometry of a point (the coordinate)
    points = vtk.vtkPoints()

    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    for i in range(ndarray.shape[0]):
        id = points.InsertNextPoint(ndarray[i,:])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        
    # Create a polydata object
    pd_points = vtk.vtkPolyData()

    # Set the points and vertices we created as the geometry and topology of the polydata
    pd_points.SetPoints(points)
    pd_points.SetVerts(vertices)

    # Write to file
    writePolydata(filename, pd_points)


# pipeline for extrating points, normals from meshes
class PointsExtractor:

  def __init__(self, num_iter=0, move_lim=0.0):

    # Read polydata from file
    self.reader = vtk.vtkPolyDataReader()

    # Clean the polydata
    self.cleanPolyData = vtk.vtkCleanPolyData()
    self.cleanPolyData.SetInputConnection(self.reader.GetOutputPort())

    # Apply smoothing
    self.smoothFilter = vtk.vtkSmoothPolyDataFilter()
    self.smoothFilter.SetInputConnection(self.cleanPolyData.GetOutputPort())
    self.smoothFilter.SetNumberOfIterations(num_iter)
    self.smoothFilter.SetRelaxationFactor(move_lim)
    self.smoothFilter.FeatureEdgeSmoothingOff  # looks weird if on
    self.smoothFilter.BoundarySmoothingOff

    # Generate normals, before subsampling so we can interpolate the pointdata
    # https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataExtractNormals
    self.normalGenerator = vtk.vtkPolyDataNormals()
    self.normalGenerator.SetInputConnection(self.smoothFilter.GetOutputPort())
    self.normalGenerator.ComputePointNormalsOn()
    self.normalGenerator.ComputeCellNormalsOff()
    self.normalGenerator.SetFeatureAngle(30.0)
    self.normalGenerator.SetSplitting(1)
    self.normalGenerator.SetConsistency(1)
    self.normalGenerator.SetAutoOrientNormals(1)  # is only valid for closed surfaces
    self.normalGenerator.SetComputePointNormals(1)
    self.normalGenerator.SetComputeCellNormals(0)
    self.normalGenerator.SetFlipNormals(0)
    self.normalGenerator.SetNonManifoldTraversal(1)        

    # Uniform sampling (adds points where mesh is too sparse)
    self.sampler = vtk.vtkPolyDataPointSampler()
    self.sampler.SetInputConnection(self.normalGenerator.GetOutputPort())

    # Interpolate normals fields (default linear interpolation)
    self.interp = vtk.vtkPointInterpolator()
    self.interp.SetInputConnection(self.sampler.GetOutputPort())
    self.interp.SetSourceConnection(self.normalGenerator.GetOutputPort())

  def extract(self, filename, target_depth):
    # return np.array of points, normals
    
    # Set filename
    self.reader.SetFileName(filename)
    
    # Get data bounds (after smoothing)
    # TODO use bounding sphere for consitency with octree builder
    self.smoothFilter.Update()
    bounds = self.smoothFilter.GetOutput().GetBounds()
    bbox_width = [bounds[3] - bounds[0],
                  bounds[4] - bounds[1],
                  bounds[5] - bounds[2]]
    
    # Estimate resolution of the octree
    # this should at least 2 points per octree leaf box
    target_dx = 0.5 * max(bbox_width)/(2**(target_depth - 1))
    self.sampler.SetDistance(target_dx)  # set target spacing
    
    # Extract points and normals (may fail if mesh is non-manifold)
    self.interp.Update()
    if self.interp.GetOutput().GetNumberOfPoints() > 0:
      points, normals = getNormals(self.interp.GetOutput())
    else:
      print("something went wrong!")
      return

    # Criteria for normal orientation: normal at max y should point in y direction
    idx_top = np.argmax(points[:, 1])  
    if np.sign(normals[idx_top, 1]) < 0.0:
      print("Flipping normals! (max y-pos point: ny=%3.2f)" % normals[idx_top, 1])
      normals = -normals

    return points, normals
