# -*- coding: utf-8 -*-
"""
Created on Thu May 05 09:37:18 2016

@author: tboom_000
"""
import numpy as np
from skimage.io import imread
from skimage.viewer import ImageViewer
from skimage.transform import resize

image = imread('Documents/Personal/Projects/Kaggle/StateFarm/source_files/img_6.jpg', 0)
small = resize(image, (120, 160))

view = ImageViewer(small)
viewer.show()

