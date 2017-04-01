import utils
import cv2
import numpy as np
from matplotlib import pylab as plt
from sklearn.preprocessing import normalize
from PIL import Image

#
try:
    #TODO: Load Raw Image
    rawimg = np.fromfile('LENA256.RAW', dtype=np.uint8, sep="")
    rawimg = rawimg.reshape((256, 256))
    img = Image.fromarray(rawimg)
    img.save('lena1.jpg')
    # imgplot = plt.imshow(rawimg)
    # plt.gray()
    # plt.colorbar()
    # plt.imsave('lena2.jpg', rawimg)
    # cv2.imwrite('lena3.jpg',rawimg)
except Exception as Argument:
    print('Exception occurred 2 {0}'.format(Argument))
else:
    plt.show()

# img_array1 = plt.imread('lena1.jpg')
# img_array2 = plt.imread('lena2.jpg')
# img_array3 = plt.imread('lena3.jpg')
# print(img_array1.shape)
# print(img_array2.shape)
# print(img_array3.shape)