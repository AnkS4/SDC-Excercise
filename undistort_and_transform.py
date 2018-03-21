import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def corners_unwarp(img, nx, ny, mtx, dist):
    # Undistort img
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    #print(corners[0], corners[7], corners[40], corners[47])
    # If corners found: 
    if ret == True:
            # a) Draw corners
            cv2.drawChessboardCorners(img, (8, 6), corners, ret)

            img_size = (gray.shape[1], gray.shape[0])
            # Define 4 source points
            src = np.float32([corners[0], corners[7], corners[40], corners[47]])
            #print(img.shape)
            offset = 100
            xmax, ymax = img.shape[1]-1, img.shape[0]-1
            # Define 4 destination points
            dst = np.float32([[offset, offset], [xmax-offset, offset], [offset, ymax-offset], [xmax-offset, ymax-offset]])
            # Get the transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp your image to a top-down view
            warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(img)
ax1.set_title('Original Image')
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image')