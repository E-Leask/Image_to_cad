# USAGE
# python detect_shapes.py --image gradient_basic.png

# import the necessary packages
#from six import u
#from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2 as cv
import numpy as np
from numpy import array
import openpyscad as ops
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt

#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler

from sympy import *

from input import *
from filter_input import *
from edge_detector import *
from line_detector import *

ratio,resized=input()
gray,filter,thresh1,grad=filter_input(resized)
erosion = edge_detect(gray, filter, grad)
line_detect(gray,erosion)



