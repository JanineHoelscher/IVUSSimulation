import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import slicer.util
from slicer.util import VTKObservationMixin
import numpy as np
import csv

#
# VesselCrossSectionAnalyzer
#
class VesselCrossSectionAnalyzer(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  [https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py]
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Vessel Cross-Section Analyzer"
    self.parent.categories = ["IVUS Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Your Name (Your Organization)"] # Replace with your name and organization
    self.parent.helpText = """
    This module analyzes a 3D vessel segmentation (STL model) along a user-defined midline curve.
    At specified intervals along the midline, it casts rays in a plane perpendicular to the midline
    tangent, measuring distances to the vessel boundary. It can output results to a CSV file
    and visualize average radii and radius variations as color-coded circles.
    """
    self.parent.acknowledgementText = """
    This module was developed by [Your Name] based on a user request and guidance from a large language model.
    """

#
# VesselCrossSectionAnalyzerWidget
#
class VesselCrossSectionAnalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  [https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py]
  """

  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets
    self.logic = VesselCrossSectionAnalyzerLogic()

    # Layout within the appropriate section of the Slicer user interface
    parametersCollapsibleBtn = ctk.ctkCollapsibleButton()
    parametersCollapsibleBtn.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleBtn)

    # Layout within the collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleBtn)

    # Input Markups Curve Node selector
    self.inputMidlineSelector = slicer.qMRMLNodeComboBox()
    self.inputMidlineSelector.nodeTypes = ["vtkMRMLMarkupsCurveNode"]
    self.inputMidlineSelector.selectNodeUponCreation = True
    self.inputMidlineSelector.addEnabled = False
    self.inputMidlineSelector.removeEnabled = False
    self.inputMidlineSelector.noneEnabled = True
    self.inputMidlineSelector.showHidden = False
    self.inputMidlineSelector.showChildNodeTypes = False
    self.inputMidlineSelector.setMRMLScene(slicer.mrmlScene)
    self.inputMidlineSelector.setToolTip("Select the midline curve (vtkMRMLMarkupsCurveNode) along the vessel.")
    parametersFormLayout.addRow("Input Midline Curve:", self.inputMidlineSelector)

    # Input STL Model Node selector
    self.inputModelSelector = slicer.qMRMLNodeComboBox()
    self.inputModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.inputModelSelector.selectNodeUponCreation = True
    self.inputModelSelector.addEnabled = False
    self.inputModelSelector.removeEnabled = False
    self.inputModelSelector.noneEnabled = True
    self.inputModelSelector.showHidden = False
    self.inputModelSelector.showChildNodeTypes = False
    self.inputModelSelector.setMRMLScene(slicer.mrmlScene)
    self.inputModelSelector.setToolTip("Select the 3D vessel segmentation model (STL/Model Node).")
    parametersFormLayout.addRow("Input Vessel Model:", self.inputModelSelector)

    # Sampling Parameters
    samplingParametersCollapsibleBtn = ctk.ctkCollapsibleButton()
    samplingParametersCollapsibleBtn.text = "Sampling Parameters"
    self.layout.addWidget(samplingParametersCollapsibleBtn)
    samplingParametersFormLayout = qt.QFormLayout(samplingParametersCollapsibleBtn)

    self.stepSizeMmSpinBox = qt.QDoubleSpinBox()
    self.stepSizeMmSpinBox.setRange(0.1, 100.0)
    self.stepSizeMmSpinBox.setValue(1.0)
    self.stepSizeMmSpinBox.setDecimals(2)
    self.stepSizeMmSpinBox.setSuffix(" mm")
    self.stepSizeMmSpinBox.setToolTip("Step size along the midline for sampling cross-sections.")
    samplingParametersFormLayout.addRow("Step Size:", self.stepSizeMmSpinBox)

    self.rayIncrementDegSpinBox = qt.QDoubleSpinBox()
    self.rayIncrementDegSpinBox.setRange(1.0, 90.0)
    self.rayIncrementDegSpinBox.setValue(10.0)
    self.rayIncrementDegSpinBox.setDecimals(1)
    self.rayIncrementDegSpinBox.setSuffix(" degrees")
    self.rayIncrementDegSpinBox.setToolTip("Angular increment for rays in the perpendicular plane.")
    samplingParametersFormLayout.addRow("Ray Increment:", self.rayIncrementDegSpinBox)

    self.maxRayLengthMmSpinBox = qt.QDoubleSpinBox()
    self.maxRayLengthMmSpinBox.setRange(1.0, 1000.0)
    self.maxRayLengthMmSpinBox.setValue(50.0)
    self.maxRayLengthMmSpinBox.setDecimals(1)
    self.maxRayLengthMmSpinBox.setSuffix(" mm")
    self.maxRayLengthMmSpinBox.setToolTip("Maximum length for rays to search for the vessel boundary. Beyond this, 'infinity' is reported.")
    samplingParametersFormLayout.addRow("Max Ray Length:", self.maxRayLengthMmSpinBox)

    # Output Options
    outputOptionsCollapsibleBtn = ctk.ctkCollapsibleButton()
    outputOptionsCollapsibleBtn.text = "Output Options"
    self.layout.addWidget(outputOptionsCollapsibleBtn)
    outputOptionsFormLayout = qt.QFormLayout(outputOptionsCollapsibleBtn)

    self.outputCsvPathSelector = ctk.ctkPathLineEdit()
    self.outputCsvPathSelector.filters = ctk.ctkPathLineEdit.Files | ctk.ctkPathLineEdit.Writable
    self.outputCsvPathSelector.nameFilters = ["CSV Files (*.csv)"]
    self.outputCsvPathSelector.setCurrentPath(os.path.join(slicer.app.temporaryPath, "VesselMeasurements.csv"))
    self.outputCsvPathSelector.setToolTip("Select the path and filename for the CSV output.")
    outputOptionsFormLayout.addRow("Output CSV File:", self.outputCsvPathSelector)

    self.visualizeCirclesCheckBox = qt.QCheckBox()
    self.visualizeCirclesCheckBox.setChecked(True)
    self.visualizeCirclesCheckBox.setToolTip("Check to visualize cross-sectional circles color-coded by radius variation.")
    outputOptionsFormLayout.addRow("Visualize Circles:", self.visualizeCirclesCheckBox)

    # Apply Button
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the analysis."
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)

    # Progress Bar
    self.progressBar = qt.QProgressBar()
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(0)
    self.progressBar.hide()
    self.layout.addWidget(self.progressBar)

    # connections
    self.applyButton.clicked.connect(self.onApplyButton)
    self.inputMidlineSelector.currentNodeChanged.connect(self.onSelect)
    self.inputModelSelector.currentNodeChanged.connect(self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    self.removeObservers()

  def onSelect(self):
    self.applyButton.enabled = self.inputMidlineSelector.currentNode() and self.inputModelSelector.currentNode()

  def onApplyButton(self):
    with slicer.util.MessageDialog("Analysis Running") as msg: #, parent = slicer.util.mainWindow()
        #msg.setCancelButton(qt.QMessageBox.Cancel)
        #msg.show()

        slicer.app.processEvents() # Process events to show the dialog
        self.progressBar.setValue(0)
        self.progressBar.show()
        slicer.app.processEvents()

        try:
            self.logic.run(
                self.inputMidlineSelector.currentNode(),
                self.inputModelSelector.currentNode(),
                self.stepSizeMmSpinBox.value,
                self.rayIncrementDegSpinBox.value,
                self.maxRayLengthMmSpinBox.value,
                self.outputCsvPathSelector.currentPath,
                self.visualizeCirclesCheckBox.isChecked(),
                self.progressBar # Pass progress bar for updates
            )
            slicer.util.delayDisplay("Analysis completed successfully!", 3000)
        except Exception as e:
            slicer.util.errorDisplay("Failed to run analysis: " + str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.progressBar.setValue(0)
            self.progressBar.hide()
            # msg.close() # Close the dialog


#
# VesselCrossSectionAnalyzerLogic
#
class VesselCrossSectionAnalyzerLogic(ScriptedLoadableModuleLogic):
  """This class implements the core logic for the module.
  """

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)

  def getVesselModelPolyData(self, modelNode):
    """
    Extracts vtkPolyData from a vtkMRMLModelNode.
    """
    if not modelNode or not modelNode.GetPolyData():
      raise ValueError("Invalid model node or polydata not found.")
    return modelNode.GetPolyData()

  def buildOBBTree(self, polyData):
    """
    Builds an OBBTree for efficient intersection testing.
    """
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(polyData)
    obbTree.BuildLocator()
    return obbTree
  
  def cross(self, vectorA, vectorB):
     '''
     returns cross product of two inputted vectors
     '''
     crossProduct = vtk.vtkVector3d()
     crossProduct = [vectorA[1]*vectorB[2]-vectorA[2]*vectorB[1], vectorA[0]*vectorB[2]-vectorA[2]*vectorB[0], vectorA[0]*vectorB[1]-vectorA[1]*vectorB[0]]
     return crossProduct
  
  def magn(self, vector):
     '''
     returns magnitude of vector
     '''
     magnitude = (vector[0]**2 + vector[1]**2 + vector[2]**2)**(1/2)
     return magnitude

  def getTangentAndPerpendicularBasis(self, curveNode, arcLengthMm, curveLengthMm, S_axis=(0.0, 0.0, 1.0), R_axis=(1.0, 0.0, 0.0)):
    """
    Calculates the tangent at a given arc length and two orthogonal basis vectors
    (U, V) for the perpendicular plane, with U aligned as much as possible with S_axis.
    :param curveNode: vtkMRMLMarkupsCurveNode
    :param arcLengthMm: Arc length in mm along the curve.
    :param S_axis: Superior-Inferior axis vector (default: (0,0,1) for LPS).
    :param R_axis: Right-Left axis vector (default: (1,0,0) for LPS), used for fallback.
    :return: A tuple (point, tangent, U_vector, V_vector).
    """
    point = [0.0, 0.0, 0.0]
    tangent = [0.0, 0.0, 0.0]

    curveNode.GetMeasurement("curvature mean").SetEnabled(True)
    positions = slicer.util.arrayFromMarkupsCurvePoints(curveNode, True)
    stepSize = round(curveLengthMm / len(positions))
    tangentArray = slicer.util.arrayFromMarkupsCurveData(curveNode, "Tangents", True)
    tangent = tangentArray[round(arcLengthMm / stepSize)]
    point = positions[round(arcLengthMm / stepSize)]

    # Convert to vtkVector3d for easier operations
    point_vec = vtk.vtkVector3d(point)
    tangent_vec = vtk.vtkVector3d(tangent)
    S_axis_vec = vtk.vtkVector3d(S_axis)
    R_axis_vec = vtk.vtkVector3d(R_axis)

    tangent_vec.Normalize()

    # Calculate U_temp = T x S_axis
    U_temp = vtk.vtkVector3d()
    U_temp = self.cross(tangent_vec, S_axis_vec)
    
    # Check for parallelism (magnitude of cross product near zero)
    if self.magn(U_temp) < 1e-6: # Tangent is parallel to S_axis
        # Fallback: cross with R_axis
        U_temp = self.cross(tangent_vec, R_axis_vec)
        if self.magn(U_temp) < 1e-6: # Tangent is also parallel to R_axis (highly unlikely, but for robustness)
            # Pick any perpendicular vector if tangent is collinear with primary axes
            vtk.vtkMath.Perpendiculars(tangent_vec, U_temp, vtk.vtkVector3d()) # U_temp becomes arbitrary perpendicular
    
    U_vector = vtk.vtkVector3d()
    U_vector = U_temp

    # Calculate V_vector = T x U_vector (makes V orthogonal to both T and U)
    V_vector = vtk.vtkVector3d()
    V_vector = self.cross(tangent_vec, U_vector)
    #V_vector.Normalize()
    
    return point_vec, tangent_vec, U_vector, V_vector

  def calculateAngleBetweenVectors(self, vec1, vec2):
    """Calculates the angle in degrees between two vtkVector3d objects."""
    dot_product = vec1.Dot(vec2)
    mag1 = self.magn(vec1)
    mag2 = self.magn(vec2)
    if mag1 == 0 or mag2 == 0:
        return np.nan # Undefined angle if one vector is zero
    
    cosine_angle = dot_product / (mag1 * mag2)
    # Clip cosine_angle to [-1, 1] to avoid numerical precision errors causing acos to return NaN
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

  def run(self, midlineCurveNode, vesselModelNode, stepSizeMm, rayIncrementDeg, maxRayLengthMm, outputCsvPath, visualizeCircles, progressBar=None):
    """
    Run the actual algorithm
    """
    if not midlineCurveNode or not vesselModelNode:
      raise ValueError("Input midline curve and vessel model are required.")

    # slicer.app.set  # Clear any previous messages
    slicer.app.processEvents() # Ensure GUI updates

    print(f"Starting analysis for curve '{midlineCurveNode.GetName()}' and model '{vesselModelNode.GetName()}'")

    vesselPolyData = self.getVesselModelPolyData(vesselModelNode)
    obbTree = self.buildOBBTree(vesselPolyData)

    curveLengthMm = midlineCurveNode.GetCurveLengthWorld()
    if curveLengthMm < stepSizeMm:
      raise ValueError("Curve length is shorter than the step size. Please increase step size or shorten curve.")

    numSteps = int(curveLengthMm / stepSizeMm)
    if numSteps == 0: # Ensure at least one step if curve is very short but longer than 0
        numSteps = 1

    all_measurements = []
    # Data structure to hold measurements for circle visualization
    cross_section_data = {}

    currentProgress = 0
    totalSteps = numSteps * (360.0 / rayIncrementDeg) # Rough estimate for progress bar

    for i in range(numSteps + 1): # +1 to include the last point even if it's not a full step
      arcLengthMm = i * stepSizeMm
      if arcLengthMm > curveLengthMm and i > 0: # Avoid going past the end of the curve for the last point
          arcLengthMm = curveLengthMm
          if i > numSteps: # Already processed the last point
              break

      # Update progress bar
      if progressBar:
          newProgress = int((arcLengthMm / curveLengthMm) * 100)
          if newProgress > currentProgress:
              currentProgress = newProgress
              progressBar.setValue(currentProgress)
              slicer.app.processEvents()

      try:
          point, tangent, U_vector, V_vector = self.getTangentAndPerpendicularBasis(midlineCurveNode, arcLengthMm, curveLengthMm)
      except Exception as e:
          print(f"Warning: Could not get tangent/basis at arc length {arcLengthMm:.2f}mm. Skipping this point. Error: {e}")
          continue

      current_section_distances = [] # For calculating avg radius and std dev for this section

      for angle_deg in np.arange(0, 360, rayIncrementDeg):
        angle_rad = np.radians(angle_deg)
        
        # Calculate ray direction in the perpendicular plane
        ray_dir = vtk.vtkVector3d()
        ray_dir.SetX(U_vector[0] * np.cos(angle_rad) + V_vector[0] * np.sin(angle_rad))
        ray_dir.SetY(U_vector[1] * np.cos(angle_rad) + V_vector[1] * np.sin(angle_rad))
        ray_dir.SetZ(U_vector[2] * np.cos(angle_rad) + V_vector[2] * np.sin(angle_rad))
        #ray_dir.Normalize()

        ray_start = [point[0], point[1], point[2]]
        ray_end = [
            point[0] + ray_dir[0] * maxRayLengthMm,
            point[1] + ray_dir[1] * maxRayLengthMm,
            point[2] + ray_dir[2] * maxRayLengthMm
        ]

        intersection_point = vtk.vtkPoints()
        #t = vtk.reference(0.0) # Parametric coordinate of intersection
        #subId = vtk.reference(0)
        cellId = vtk.reference(0)
        #pCoords = [0.0, 0.0, 0.0]

        # Perform ray intersection
        num_intersects = obbTree.IntersectWithLine(ray_start, ray_end, intersection_point, None)
        
        print(f"intersectionpoint = {intersection_point}")

        distance = float('inf')
        int_x, int_y, int_z = np.nan, np.nan, np.nan
        normal_x, normal_y, normal_z = np.nan, np.nan, np.nan
        angle_ray_normal_deg = np.nan

        if num_intersects > 0:
          # Intersection occurred
          distance = np.linalg.norm(np.array(intersection_point) - np.array(ray_start))
          int_x, int_y, int_z = intersection_point[0], intersection_point[1], intersection_point[2]
          
          # Get surface normal at intersection point
          intersected_cell = vesselPolyData.GetCell(cellId.get())
          if intersected_cell and intersected_cell.GetCellType() == vtk.VTK_TRIANGLE: # Ensure it's a triangle
            p0 = vesselPolyData.GetPoint(intersected_cell.GetPointId(0))
            p1 = vesselPolyData.GetPoint(intersected_cell.GetPointId(1))
            p2 = vesselPolyData.GetPoint(intersected_cell.GetPointId(2))
            
            # Calculate face normal
            v1 = vtk.vtkVector3d(p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
            v2 = vtk.vtkVector3d(p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
            
            face_normal = vtk.vtkVector3d()
            face_normal = self.cross(v1, v2)
            face_normal.Normalize()

            normal_x, normal_y, normal_z = face_normal.GetX(), face_normal.GetY(), face_normal.GetZ()
            
            # Calculate angle between ray and normal
            angle_ray_normal_deg = self.calculateAngleBetweenVectors(ray_dir, face_normal)

          current_section_distances.append(distance)


        # Store all raw measurements
        all_measurements.append({
            "SampleID": i,
            "ArcLength_mm": arcLengthMm,
            "Midline_X": point.GetX(),
            "Midline_Y": point.GetY(),
            "Midline_Z": point.GetZ(),
            "Tangent_X": tangent.GetX(),
            "Tangent_Y": tangent.GetY(),
            "Tangent_Z": tangent.GetZ(),
            "U_X": U_vector[0],
            "U_Y": U_vector[1],
            "U_Z": U_vector[2],
            "V_X": V_vector[0],
            "V_Y": V_vector[1],
            "V_Z": V_vector[2],
            "RayAngle_Deg": angle_deg,
            "RayDir_X": ray_dir.GetX(),
            "RayDir_Y": ray_dir.GetY(),
            "RayDir_Z": ray_dir.GetZ(),
            "Intersection_X": int_x,
            "Intersection_Y": int_y,
            "Intersection_Z": int_z,
            "Distance_mm": distance,
            "Normal_X": normal_x,
            "Normal_Y": normal_y,
            "Normal_Z": normal_z,
            "AngleRayNormal_Deg": angle_ray_normal_deg
        })
      
      # Store data for circle visualization for this cross-section
      if current_section_distances:
          avg_radius = np.mean(current_section_distances)
          std_dev_radius = np.std(current_section_distances)
          cross_section_data[i] = {
              "point": point,
              "U_vector": U_vector,
              "V_vector": V_vector,
              "avg_radius": avg_radius,
              "std_dev_radius": std_dev_radius
          }
      else:
          cross_section_data[i] = { # Store nan if no intersections found
              "point": point,
              "U_vector": U_vector,
              "V_vector": V_vector,
              "avg_radius": np.nan,
              "std_dev_radius": np.nan
          }

    # Write to CSV
    self.writeMeasurementsToCsv(outputCsvPath, all_measurements)

    # Visualize circles
    if visualizeCircles:
        self.visualizeCrossSectionCircles(cross_section_data)

    print("Analysis complete.")
    if progressBar:
        progressBar.setValue(100)
        slicer.app.processEvents()


  def writeMeasurementsToCsv(self, outputPath, measurements):
    """
    Writes the collected measurements to a CSV file.
    """
    if not measurements:
      print("No measurements to write to CSV.")
      return

    keys = measurements[0].keys()
    try:
      with open(outputPath, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(measurements)
      print(f"Measurements saved to: {outputPath}")
    except IOError as e:
      print(f"Error writing CSV file: {e}")
      raise IOError(f"Could not write CSV to {outputPath}: {e}")

  def visualizeCrossSectionCircles(self, crossSectionData):
    """
    Generates and visualizes cross-sectional circles color-coded by radius variation.
    """
    print("Generating circle visualization...")

    if not crossSectionData:
      print("No cross-section data available for visualization.")
      return

    # Create a single model node for all circles
    circleModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "VesselCrossSectionCircles")
    circlePolyData = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()
    colors = vtk.vtkFloatArray() # For scalar coloring (std dev)
    colors.SetName("RadiusVariation")
    
    # Define a colormap for standard deviation
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableValue(0, 0.0, 0.0, 1.0) # Blue for low std dev
    lut.SetTableValue(1, 1.0, 0.0, 0.0) # Red for high std dev
    lut.Build()

    # Determine min/max std_dev for colormapping
    all_std_devs = [data["std_dev_radius"] for data in crossSectionData.values() if not np.isnan(data["std_dev_radius"])]
    if not all_std_devs:
        print("No valid standard deviations to color code. Skipping visualization.")
        slicer.mrmlScene.RemoveNode(circleModelNode)
        return

    min_std_dev = min(all_std_devs)
    max_std_dev = max(all_std_devs)
    if min_std_dev == max_std_dev: # Avoid division by zero for uniform std dev
        max_std_dev += 1e-6 # Add a small epsilon

    for sample_id, data in crossSectionData.items():
        point = data["point"]
        U_vector = data["U_vector"]
        V_vector = data["V_vector"]
        avg_radius = data["avg_radius"]
        std_dev_radius = data["std_dev_radius"]

        if np.isnan(avg_radius) or np.isnan(std_dev_radius):
            continue # Skip if no valid measurements for this section

        num_segments = 36 # For creating a smooth circle (36 segments for 10 degree increments)
        current_point_id_offset = points.GetNumberOfPoints() # Store current number of points before adding new ones

        # Create circle points
        for i in range(num_segments):
            angle_rad = 2 * np.pi * i / num_segments
            circle_x = point.GetX() + avg_radius * (U_vector.GetX() * np.cos(angle_rad) + V_vector.GetX() * np.sin(angle_rad))
            circle_y = point.GetY() + avg_radius * (U_vector.GetY() * np.cos(angle_rad) + V_vector.GetY() * np.sin(angle_rad))
            circle_z = point.GetZ() + avg_radius * (U_vector.GetZ() * np.cos(angle_rad) + V_vector.GetZ() * np.sin(angle_rad))
            points.InsertNextPoint(circle_x, circle_y, circle_z)
            
            # Map standard deviation to scalar value [0, 1] for colormapping
            normalized_std_dev = (std_dev_radius - min_std_dev) / (max_std_dev - min_std_dev)
            colors.InsertNextValue(normalized_std_dev)

        # Create circle polygon (closed loop)
        for i in range(num_segments):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, current_point_id_offset + i)
            line.GetPointIds().SetId(1, current_point_id_offset + (i + 1) % num_segments)
            polygons.InsertNextCell(line)

    circlePolyData.SetPoints(points)
    circlePolyData.SetLines(polygons) # Using lines to draw circles
    circlePolyData.GetPointData().SetScalars(colors) # Assign colors to points

    circleModelNode.SetAndObservePolyData(circlePolyData)

    # Set display properties
    displayNode = circleModelNode.GetDisplayNode()
    if not displayNode:
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        circleModelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    
    displayNode.SetScalarVisibility(True)
    displayNode.SetScalarRangeFlagToAutomatic()
    displayNode.SetActiveScalar("RadiusVariation", vtk.vtkAssignAttribute.POINT_DATA)
    displayNode.SetColorMap(lut)
    displayNode.SetAndObserveColorNodeID(slicer.util.getNode('HotToCold').GetID()) # A good default colormap
    displayNode.SetSliceDisplayModeToVisibility(slicer.vtkMRMLSliceDisplayNode.Visibility3D)
    displayNode.SetClippingEnabled(False)
    displayNode.SetOpacity(0.8)
    displayNode.SetRepresentation(displayNode.Wireframe) # Wireframe for circles
    displayNode.SetPointSize(3) # Make points visible if desired
    displayNode.SetAmbient(0.4)
    displayNode.SetDiffuse(0.6)
    displayNode.SetSpecular(0.0)
    displayNode.SetLighting(True)


    print("Circle visualization created successfully: VesselCrossSectionCircles")


#
# VesselCrossSectionAnalyzerTest
#
class VesselCrossSectionAnalyzerTest(ScriptedLoadableModuleTest):
  """
  This is the test class for the scripted loadable module.
  """

  def setUp(self):
    """ Clear scene before each test
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """
    Run as few VIP tests as possible during module development.
    """
    self.setUp()
    self.test_VesselCrossSectionAnalyzer1()

  def test_VesselCrossSectionAnalyzer1(self):
    """
    A sample test case for a simple scenario.
    """
    self.delayDisplay("Starting the test")

    # Load test data
    # Create a simple straight midline curve
    curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "TestMidline")
    curvePoints = vtk.vtkPoints()
    curvePoints.InsertNextPoint(0, 0, 0)
    curvePoints.InsertNextPoint(0, 0, 100) # 100mm straight line along Z
    curveNode.SetControlPointPositionsWorld(curvePoints)

    # Create a simple cylindrical vessel model (STL)
    # This is a placeholder; in a real scenario, you'd load an actual STL.
    # We'll create a basic VTK cylinder.
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetHeight(150) # Make it longer than the curve
    cylinder.SetRadius(10) # 10mm radius
    cylinder.SetResolution(50)
    cylinder.Update()
    
    transform = vtk.vtkTransform()
    transform.Translate(0.0, 0.0, 50.0) # Center cylinder around curve
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(cylinder.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TestVesselModel")
    modelNode.SetAndObservePolyData(transformFilter.GetOutput())

    self.delayDisplay("Test data loaded: Midline and Cylinder Model")

    logic = VesselCrossSectionAnalyzerLogic()

    # Define test parameters
    stepSizeMm = 5.0
    rayIncrementDeg = 20.0
    maxRayLengthMm = 20.0
    outputCsvPath = os.path.join(slicer.app.temporaryPath, "TestVesselMeasurements.csv")
    visualizeCircles = True

    # Run the analysis
    print("Running analysis with test parameters...")
    logic.run(curveNode, modelNode, stepSizeMm, rayIncrementDeg, maxRayLengthMm, outputCsvPath, visualizeCircles)

    self.delayDisplay("Analysis completed for test case. Checking results...")

    # Basic assertions to check if output files/nodes were created
    self.assertTrue(os.path.exists(outputCsvPath))
    print(f"CSV output exists at: {outputCsvPath}")

    # Check if visualization model node was created
    circlesModelNode = slicer.util.getNode("VesselCrossSectionCircles")
    self.assertIsNotNone(circlesModelNode)
    print("VesselCrossSectionCircles model node created.")
    self.assertGreater(circlesModelNode.GetPolyData().GetNumberOfCells(), 0) # Should have circles drawn

    # Optional: Read CSV and do more detailed checks
    with open(outputCsvPath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        self.assertGreater(len(rows), 0)
        # Check a few key columns for expected values
        self.assertIn('Distance_mm', rows[0])
        self.assertIn('RayAngle_Deg', rows[0])
        # For a cylinder, most distances should be close to the radius (10mm)
        # Check a few distance values
        for row in rows[:10]: # Check first 10 rows
            dist = float(row['Distance_mm'])
            # Allow some tolerance for floating point math, ideally should be 10.0
            self.assertAlmostEqual(dist, 10.0, delta=0.5) 


    self.delayDisplay('Test passed!')

