#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2017 - Limber Cheng <cheng@limberence.com> 
# @Time : 21/03/2018 22:13
# @Author : Limber Cheng
# @File : laneline
# @Software: PyCharm
from glob import glob

import cv2
import matplotlib.image as mpimg
import numpy as np

# Load in the chessboard calibration images to a list
cal_image_loc = glob('camera_cal/*.jpg')
calibration_images = []

for fname in cal_image_loc:
    img = mpimg.imread(fname)
    calibration_images.append(img)

# Prepare object points
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays for later storing object points and image points
objpoints = []
imgpoints = []

# Iterate through images for their points
for image in calibration_images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(image, (9, 6), corners, ret)

# Returns camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)




