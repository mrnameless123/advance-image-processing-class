import numpy as np

from skimage.transform import (probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm


# Constructing test image
# image = np.zeros((100, 100))
# idx = np.arange(25, 75)
# image[idx[::-1], idx] = 255
# image[idx, idx] = 255

# # Classic straight-line Hough transform
# h, theta, d = hough_line(image)
#
# # Generating figure 1
# fig, axes = plt.subplots(1, 3, figsize=(15, 6),
#                          subplot_kw={'adjustable': 'box-forced'})
# ax = axes.ravel()
#
# ax[0].imshow(image, cmap=cm.gray)
# ax[0].set_title('Input image')
# ax[0].set_axis_off()
#
# ax[1].imshow(np.log(1 + h),
#              extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
#              cmap=cm.gray, aspect=1/1.5)
# ax[1].set_title('Hough transform')
# ax[1].set_xlabel('Angles (degrees)')
# ax[1].set_ylabel('Distance (pixels)')
# ax[1].axis('image')
#
# ax[2].imshow(image, cmap=cm.gray)
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#     y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
#     ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
# ax[2].set_xlim((0, image.shape[1]))
# ax[2].set_ylim((image.shape[0], 0))
# ax[2].set_axis_off()
# ax[2].set_title('Detected lines')
#
# plt.tight_layout()
# plt.show()

# Line finding using the Probabilistic Hough Transform
# image = data.camera()
# edges = canny(image, 2, 1, 25)
from PIL import Image
from scipy import signal as sg
import utils
threshold_upper = 80
threshold_lower = 80
str_image_name = 'Lena'
max_dist = 1.5
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
image = Image.open(str_image_name + '.jpg')
image = np.array(image)
vertical_edge = sg.convolve2d(image, kernel_x, mode = 'same')
horizontal_edge = sg.convolve2d(image, kernel_y, mode = 'same')
edge_image = vertical_edge+horizontal_edge
edge_image = utils.func_verify_image(edge_image, threshold_1 = threshold_upper, threshold_2=threshold_lower)
lines = probabilistic_hough_line(edge_image, threshold=10, line_length=10,
                                 line_gap=1)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edge_image, cmap=cm.gray)
ax[1].set_title('Sobel edges')

ax[2].imshow(edge_image * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()
    a.set_adjustable('box-forced')

plt.tight_layout()
plt.show()