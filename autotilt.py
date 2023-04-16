from math import *
import numpy as np
from ncempy.io import dm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy import optimize
from ipywidgets import widgets
import warnings
import itertools
warnings.filterwarnings('ignore')

#Declare Global Variables
xPeaks = np.array([])
yPeaks = np.array([])
#Class that represents lines in point, slope form
class Line:
    def __init__(self, xPoint, yPoint, slope):
        self.xPoint = xPoint
        self.yPoint = yPoint
        self.slope = slope
 
    def setColor(self, color):
        self.color = color
 
    def getX(self):
        return self.xPoint

    def getY(self):
        return self.yPoint

    def getSlope(self):
        return self.slope

#Finds the midpoint between two points
def midpoint(point1, point2):
  xMid = (point1[0] + point2[0])/2
  yMid = (point1[1] + point2[1])/2
  return xMid, yMid

#Finds the slope of the line perpendicular a line segment
def perpendicularSlope(point1, point2):
  slope = (point2[1] - point1[1])/(point2[0] - point1[0])
  perpSlope = -1/slope
  return perpSlope

#Finds the intersection of two lines given their point, slope formulas
def findCoords(line1,line2):
  m1 = line1.getSlope()
  x1 = line1.getX()
  y1 = line1.getY()
  m2 = line2.getSlope()
  x2 = line2.getX()
  y2 = line2.getY()
  xCoord = (m1 * x1 - y1 - m2 * x2 + y2)/(m1 - m2)
  yCoord = m1 * (xCoord- x1) + y1
  return xCoord, yCoord

#Code adapted from: http://www.scipy.org/Cookbook/Least_Squares_Circle
#Calculates the radius of the circle
def calc_R(x, y):
    global xPeaks
    global yPeaks
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((xPeaks-x)**2 + (yPeaks-y)**2)

#plots the circle given the center and the radius
def plot(x, y, R):
    global xPeaks
    global yPeaks
    f = plt.figure( facecolor='white')
    plt.axis('equal')

    theta_fit = np.linspace(-pi, pi, 180)

    x_fit = x + R*np.cos(theta_fit)
    y_fit = y + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit)
    plt.plot(x, y, 'gD')

    # draw
    plt.draw()

    # plot data
    plt.plot(xPeaks, yPeaks, 'ro', mec='b')

def f(c):
  #  """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

#Finds the peak of each diffraction spot that is at least a certain cluster size
#User Defined Minimum Pixel Cluster Size
def createPeaks(clusterSize):
  peaks = np.array([])
  for i in range(1,num_features+1):
    if (labeled_array == i).sum() >= clusterSize:
      peaks = np.append(peaks,np.unravel_index((diff*(labeled_array==i)).argmax(),diff.shape))
  peaks = np.flip(np.reshape(peaks,(-1,2)))
  return peaks
  
# You need to grant access to your google drive each time
from google.colab import drive
drive.mount('/content/drive')

# Where the file is saved, I changed the folder name to a simple one, you need to do the same
folder = '/content/drive/My Drive/William/'

#Peak Hunting Algorithm from https://journals.iucr.org/j/issues/2013/06/00/nb5079/index.html
img = dm.dmReader(folder+'06.dm3')
data = img['data']
pixelsize = np.array(img['pixelSize'])
(ny, nx) = data.shape
xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

#Creates the background frame 
#User-Defined filter, typically more than 10 pixels
filter = 50
background = gaussian_filter(data, sigma = filter)
(bgny, bgnx) = background.shape
bgxx, bgyy = np.meshgrid(np.arange(bgnx), np.arange(bgny))

#Creates the diffraction frame
#User-Defined filter, typically from 0.5 - 2
filter = 1;
diffraction = gaussian_filter(data, sigma = filter)
(dny, dnx) = diffraction.shape
dxx, dyy = np.meshgrid(np.arange(dnx), np.arange(dny))

#Finds pixels that belong to a diffraction spot
#User-Defined threshold
threshold = 100
diff = diffraction - background
diff = np.where(diff < threshold, 0, diff)

#Finds the number of diffraction spots that exist
labeled_array, num_features = ndimage.label(diff, np.ones((3,3)))

#Method 1 Uses Circle Theorem
#Any perpendicualr bisector of a chord of a circle travels through the center
#The intersection of two perpendicular bisectors can approximate the center

def findPossibleCenters(scaling,peaks):
  slopes = np.array([])
  lines = np.array([])

  #A line segment connecting two peaks is likely an approximate chord of the circle
  #Finds the perpendicular bisector of each possible chord approximation
  for data in itertools.combinations(peaks, 2):
    mid = midpoint(data[0], data[1])
    slope = perpendicularSlope(data[0], data[1])
    slopes = np.append(slopes,slope)
    lines = np.append(lines,(Line(mid[0],mid[1],slope)))

  #Creates equations for finding the intersection of two perpendicular bisectors
  lineSystems = np.array([])
  for systems in itertools.combinations(lines, 2):
      lineSystems = np.append(lineSystems,systems)
  lineSystems = np.reshape(lineSystems,(-1,2))

  #Solves and finds the intersection points of all the above equations
  possibleCenters = np.array([])
  for systems in lineSystems:
    possibleCenters = np.append(possibleCenters,findCoords(systems[0],systems[1]))

  possibleCenters = np.reshape(possibleCenters,(-1,2))
  xPoss = np.array([sub[0] for sub in possibleCenters])
  yPoss = np.array([sub[1] for sub in possibleCenters])

  #Removes outliers with respect to a user-defined scaling factor
  while 1:
    dist_2_center = ((possibleCenters - possibleCenters.mean(0)) ** 2).sum(1)
    threshold = dist_2_center.std() * scaling
    #print(possibleCenters.shape)
    #print(possibleCenters[dist_2_center<threshold].shape)
    if(possibleCenters.shape[0] == possibleCenters[dist_2_center<threshold].shape[0]):
      break
    possibleCenters = possibleCenters[dist_2_center<threshold]

  xPoss = np.array([sub[0] for sub in possibleCenters])
  yPoss = np.array([sub[1] for sub in possibleCenters])

  return xPoss,yPoss

#GUI for drawing the Laue Circle
#Cluster Size: Minimum number of pixels required to be recognized as a diffraction spot
#Scaling Factor: For Method 1, the scaling factor is used to remove outliers
cluster_input = widgets.IntText(value = 10, description='Cluster Size:')
display(cluster_input)
scaling_input = widgets.FloatText(value = 4, description = 'Scaling')
display(scaling_input)

button = widgets.Button(description="Draw Laue Circle")
output = widgets.Output()

display(button, output)

def on_button_clicked(b):
    global xPeaks
    global yPeaks
    with output:
        output.clear_output()
        #Displays the Peaks of the image
        peaks = createPeaks(cluster_input.value)
        xPeaks = np.array([sub[0] for sub in peaks])
        yPeaks = np.array([sub[1] for sub in peaks])
        plt.plot(xPeaks,yPeaks,"k.")
        plt.title('Peak Display')
        plt.show()

        #Method 1: Uses Perpendicular Bisector Of A Chord Circle Theorem
        #Averages all of the possible centers
        xPoss,yPoss = findPossibleCenters(scaling_input.value,peaks)
        RiGeo       = calc_R(np.mean(xPoss), np.mean(yPoss))
        RGeo        = RiGeo.mean()
        xGeo        = np.mean(xPoss)
        yGeo        = np.mean(yPoss)
        plot(xGeo,yGeo, RGeo)
        plt.title('Method 1: Geometric Solution')
        plt.show()
        print("Center: ("+str(xGeo)+","+str(yGeo)+")")
        print("Radius: "+str(RGeo))
        print()
        
        #Method 2: Uses optimize.leastsq
        #Code adapted from: http://www.scipy.org/Cookbook/Least_Squares_Circle
        center_estimate = np.mean(xPeaks), np.mean(yPeaks)
        center, ier = optimize.leastsq(f, center_estimate)
        xLeastSQ, yLeastSQ = center
        RiLeastSQ       = calc_R(xLeastSQ, yLeastSQ)
        RLeastSQ        = RiLeastSQ.mean()
        plot(xLeastSQ, yLeastSQ, RLeastSQ)
        plt.title('Method 2: Least Squares Solution')
        plt.show()
        print("Center: ("+str(xLeastSQ)+","+str(yLeastSQ)+")")
        print("Radius: "+str(RLeastSQ))
button.on_click(on_button_clicked)
