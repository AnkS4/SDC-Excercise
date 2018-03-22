import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

img = mpimg.imread('signs_vehicles_xygrad.png') # Reads in RGB format
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert to gray 

k = 15 # Set Odd Kernel Size, larger to smooth gradient measurements
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k) # Derivative in X direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k) # Derivative in Y direction
abs_sobelx = np.absolute(sobelx) #Absolute value of X derivative
abs_sobely = np.absolute(sobely) #Absolute value of Y derivative

# Outputs absolute threshold value respective to the passed orientation
def abs_sobel_thresh(abs_sobelx, abs_sobely, orient='x', thresh=(0, 255)):
	if orient == 'x':
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	if orient == 'y':
		scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

	grad_binary = np.zeros_like(scaled_sobel)
	grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	return grad_binary

# Outputs magnitude of the gradient in the given threshold
def mag_thresh(sobelx, sobely, thresh=(0, 255)):
	abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)

	scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

	mag_binary = np.zeros_like(scaled_sobel)
	mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	return mag_binary

# Outputs direction of the gradient in the given threshold
def dir_threshold(abs_sobelx, abs_sobely, thresh=(0, np.pi/2)):
	grad_dir = np.arctan2(abs_sobely, abs_sobelx)

	dir_binary = np.zeros_like(grad_dir)
	dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
	return dir_binary

# Call the respective functions
gradx = abs_sobel_thresh(abs_sobelx, abs_sobely, orient='x', thresh=(50, 255))
grady = abs_sobel_thresh(abs_sobelx, abs_sobely,  orient='y', thresh=(50, 255))
mag_binary = mag_thresh(sobelx, sobely, thresh=(50, 255))
dir_binary = dir_threshold(abs_sobelx, abs_sobely, thresh=(0.7, 1.3))

# Selects the pixels where the X, Y gradients, magnitude and direction are within their threshold values.
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1

#Plot all the outputs
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image')
ax2.imshow(gradx, cmap='gray')
ax2.set_title('Absolute Gradient X')
ax3.imshow(grady, cmap='gray')
ax3.set_title('Absolute Gradient Y')
ax4.imshow(mag_binary, cmap='gray')
ax4.set_title('Magnitude of Gradient')
ax5.imshow(dir_binary, cmap='gray')
ax5.set_title('Direction of Gradient')
ax6.imshow(combined, cmap='gray')
ax6.set_title('Combined Threshold')
plt.show()