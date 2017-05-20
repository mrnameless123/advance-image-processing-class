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
    gaussian_image2 = utils.func_add_noisy(img_array, var=75)
    pepper_image1 = utils.func_add_noisy(img_array, 's&p',amount=0.01)

except Exception as Argument:
    print('Adding Gaussian noise exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(gaussian_image1).save('Gaussian_img_1.jpg')
    Image.fromarray(pepper_image1).save('Salt&pepper_img_2.jpg')

    print('Successfully Added Gaussian Noise')



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
# rotate_image12 = Image.fromarray(img_array).rotate(45, expand=True, resample=None).save('rotate_img_12.jpg')
rotate_image13 = Image.fromarray(img_array).rotate(45, expand=True, resample=Image.BILINEAR).save('rotate_img_13.jpg')
rotate_image13 = Image.fromarray(img_array).rotate(10, expand=True, resample=Image.BILINEAR).save('rotate_img_14.jpg')
try:
    rotate_image4 = utils.func_manual_rotate_image_interpolation(img_array, 45, 1)
except Exception as Argument:
    print('Adding Laplacian noise exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(utils.func_verify_image(rotate_image4)).save('rotate_img_4.jpg')
    print('Successfully Rotated Image')