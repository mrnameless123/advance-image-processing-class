import utils
import numpy as np
from matplotlib import pylab as plt
from PIL import Image
import cv2

#TODO: Load image into an array
raw_img = np.fromfile('LENA256.RAW', dtype=np.uint8, sep="")
img_array = raw_img.reshape((256, 256))

'''
Add Gaussian noise
'''
try:
    gaussian_image1 = utils.func_add_noisy(img_array)
    Image.fromarray(gaussian_image1).save('Gaussian_img_1.jpeg')
    gaussian_image2 = utils.func_add_noisy(img_array, mean=0.2)
    Image.fromarray(gaussian_image2).save('Gaussian_img_2.jpeg')
except Exception as Argument:
    print('Adding Gaussian noise exception occurred: {0}'.format(Argument))
    input()
else:
    print('Added gaussian noise successfully')



