#-*- coding: utf-8 -*-
import matplotlib.delaunay as md
import numpy as np
from scipy import ndimage
from pylab import *
from PIL import Image
from scipy import linalg
import camera

K = array([[1000,0,500],[0,1000,300],[0,0,1]])
tmp = camera.rotation_matrix([0,0,1])[:3,:3]
Rt = hstack((tmp,array([[50],[40],[30]])))
cam = camera.Camera(dot(K,Rt))

print K,Rt
print cam.factor()
