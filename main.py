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
    gaussian_image2 = utils.func_add_noisy(img_array, mean=0.2)
    gaussian_image3 = utils.func_add_noisy(img_array, mean=0.3, var=0.05)
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
try:
    laplacian_image1 = utils.func_add_noisy(img_array, noise_typ='laplacian')
    laplacian_image2 = utils.func_add_noisy(img_array, noise_typ='laplacian', mean=0.2, exponential_decay= 0.5)
    laplacian_image3 = utils.func_add_noisy(img_array, noise_typ='laplacian', mean=0.1, exponential_decay= 0.15)
except Exception as Argument:
    print('Adding Laplacian noise exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(laplacian_image1).save('Laplacian_img_1.jpg')
    Image.fromarray(laplacian_image2).save('Laplacian_img_2.jpg')
    Image.fromarray(laplacian_image3).save('Laplacian_img_3.jpg')
    print('Successfully Added Laplacian Noise')

'''
    Rotate image by interpolation
'''
def func_manual_rotate_image_interpolation(param_input, param_angle):
    try:
        rows, cols = np.array(param_input).shape
        ratio = (param_angle / 180) * np.pi
        origin_center = np.array([np.round(rows / 2), np.round(cols / 2)])
        # TODO: the reflected image b[k][l] = a[N-k][l] page17chap1
        kernel = np.array([[np.cos(-ratio), np.sin(-ratio)], [-np.sin(-ratio), np.cos(-ratio)]])
        new_row = np.int(np.abs(cols * np.cos(ratio)) + np.abs(rows * np.sin(ratio)))
        new_col = np.int(np.abs(cols * np.sin(ratio)) + np.abs(rows * np.cos(ratio)))
        output = np.zeros((new_row, new_col))
        output_center = np.array([np.round(new_row / 2), np.round(new_col / 2)])
        for x in range(new_row):
            for y in range(new_col):
                vector_rotate = np.array([x - output_center[0], y - output_center[1]])
                tmp = np.dot(kernel.T, vector_rotate)
                rotate_momentum = np.array(tmp + origin_center, dtype=np.int)
                if 0 < rotate_momentum[0]  and  rotate_momentum[0] < rows and 0 < rotate_momentum[1] and rotate_momentum[1] < cols:
                    output[x][y] = param_input[rotate_momentum[0]][rotate_momentum[1]]
    except Exception as Argument:
        print('func_manual_rotate_image_interpolation exception occurred: {0}'.format(Argument))
        input()
    else:
        return output

try:
    rotate_image1 = func_manual_rotate_image_interpolation(img_array, 45)
    rotate_image2 = func_manual_rotate_image_interpolation(img_array, 60)
except Exception as Argument:
    print('Adding Laplacian noise exception occurred: {0}'.format(Argument))
    input()
else:
    Image.fromarray(utils.func_verify_image(rotate_image1)).save('rotate_img_1.jpg')
    Image.fromarray(utils.func_verify_image(rotate_image2)).save('rotate_img_2.jpg')
    print('Successfully Rotated Image')



