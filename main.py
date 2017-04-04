import utils
import numpy as np
from scipy import ndimage
from matplotlib import pylab as plt
from PIL import Image
import cv2

#TODO: Load image into an array
raw_img = np.fromfile('LENA256.RAW', dtype=np.uint8, sep="")
img_array = raw_img.reshape((256, 256))
# raw_img = np.fromfile('Ecoli exp2.png',dtype=np.uint8, sep="")
# img_array = raw_img.reshape((640, 387))
'''
Add Gaussian noise
'''
try:
    gaussian_image1 = utils.func_add_noisy(img_array)
    gaussian_image2 = utils.func_add_noisy(img_array, mean=0.20, var=0.200)
    #Analysis RMS of noise
    gaussian_image3 = utils.func_add_noisy(img_array, mean=0.3, var=0.01)
except Exception as Argument:
    print('Adding Gaussian noise exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(gaussian_image1).save('Gaussian_img_1.jpg')
    Image.fromarray(gaussian_image2).save('Gaussian_img_2.jpg')
    Image.fromarray(gaussian_image3).save('Gaussian_img_3.jpg')
    print('Successfully Added Gaussian Noise')

'''
Add Laplace noise
'''
# try:
#     laplacian_image1 = utils.func_add_noisy(img_array, noise_typ='laplacian')
#     laplacian_image2 = utils.func_add_noisy(img_array, noise_typ='laplacian', mean=0.2, exponential_decay= 0.5)
#     laplacian_image3 = utils.func_add_noisy(img_array, noise_typ='laplacian', mean=0.1, exponential_decay= 0.15)
# except Exception as Argument:
#     print('Adding Laplacian noise exception occurred: {0}'.format(Argument))
#     input()
# else:
#     Image.fromarray(laplacian_image1).save('Laplacian_img_1.jpg')
#     Image.fromarray(laplacian_image2).save('Laplacian_img_2.jpg')
#     Image.fromarray(laplacian_image3).save('Laplacian_img_3.jpg')
#     print('Successfully Added Laplacian Noise')

'''
    Rotate image by interpolation
'''

try:
    rotate_image1 = utils.func_manual_rotate_image_interpolation(img_array, 45)
    rotate_image2 = utils.func_manual_rotate_image_interpolation(img_array, 10)
    rotate_image3 = utils.func_manual_rotate_image_interpolation(img_array, 90)
except Exception as Argument:
    print('Rotating image exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(utils.func_verify_image(rotate_image1)).save('rotate_img_1.jpg')
    Image.fromarray(utils.func_verify_image(rotate_image2)).save('rotate_img_2.jpg')
    Image.fromarray(utils.func_verify_image(rotate_image3)).save('rotate_img_3.jpg')
    #Use build in function
#     rotate_image12 = Image.fromarray(img_array).rotate(45, expand=True, resample=Image.NEAREST).save('rotate_img_12.jpg')
#     rotate_image13 = Image.fromarray(img_array).rotate(45, expand=True, resample=Image.BILINEAR).save('rotate_img_13.jpg')
#     print('Successfully Rotated Image')
rotate_image12 = Image.fromarray(img_array).rotate(45, expand=True, resample=Image.NEAREST).save('rotate_img_12.jpg')
rotate_image13 = Image.fromarray(img_array).rotate(45, expand=True, resample=Image.BILINEAR).save('rotate_img_13.jpg')
try:
    rotate_image4 = utils.func_manual_rotate_image_interpolation(img_array, 45, 1)
except Exception as Argument:
    print('Adding Laplacian noise exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(utils.func_verify_image(rotate_image4)).save('rotate_img_4.jpg')
    print('Successfully Rotated Image')