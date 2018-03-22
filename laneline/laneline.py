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

class Line:
    """
    Define a class to receive the characteristics of each line detection
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # recent polynomial coefficients
        self.recent_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # counter to reset after 5 iterations if issues arise
        self.counter = 0

class ImageProcess:
    def __init__(self, img, s_thresh=(125, 255), sx_thresh=(10, 100), R_thresh=(200, 255), sobel_kernel=3):
        self.img = img
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.R_thresh = R_thresh
        self.sobel_kernel = sobel_kernel

    def binary_images(self):
        """
        Pipeline to create binary image.
        This version uses thresholds on the R & S color channels and Sobelx.
        Binary activation occurs where any two of the three are activated.
        """
        distorted_img = np.copy(self.img)
        dst = cv2.undistort(distorted_img, mtx, dist, None, mtx)
        # Pull R
        R = dst[:, :, 0]
        # Convert to HLS colorspace
        hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Sobelx - takes the derivate in x, absolute value, then rescale
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(scaled_sobelx >= self.sx_thresh[0]) & (scaled_sobelx <= self.sx_thresh[1])] = 1

        # Threshold R color channel
        R_binary = np.zeros_like(R)
        R_binary[(R >= self.R_thresh[0]) & (R <= self.R_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        # If two of the three are activated, activate in the binary image
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[((s_binary == 1) & (sxbinary == 1)) | ((sxbinary == 1) & (R_binary == 1))
                        | ((s_binary == 1) & (R_binary == 1))] = 1

        return combined_binary

    def birds_eye(self, mtx, dist):
        """ Birds eye first undistorts the image, using the calibration from earlier.
        Next, using defined source image points and destination points,
        it will transform the image as if the road was viewed from above,
        like a bird would see. Returns the birds eye image and transform matrix.
        """
        # Put the image through the pipeline to get the binary image
        binary_img = self.binary_images(self.img)

        # Undistort
        undist = cv2.undistort(binary_img, mtx, dist, None, mtx)

        # Grab the image shape
        img_size = (undist.shape[1], undist.shape[0])

        # Source points - defined area of lane line edges
        src = np.float32([[690, 450], [1110, img_size[1]], [175, img_size[1]], [595, 450]])

        # 4 destination points to transfer
        offset = 300  # offset for dst points
        dst = np.float32([[img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]],
                          [offset, img_size[1]], [offset, 0]])

        # Use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Use cv2.warpPerspective() to warp the image to a top-down view
        top_down = cv2.warpPerspective(undist, M, img_size)
        return top_down, M

