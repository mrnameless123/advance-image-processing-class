import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def hough(im):
    """
    :param im: is a binary edge image 
    :return: a result matrix of hough transform
    """
    if isinstance(im, np.ndarray):
        clone_im = im
        im_x, im_y = im.shape
    else:
        clone_im = im.load()
        im_x, im_y = im.size
    ntx, mry = im_x, im_y
    mry = np.int(mry/2)*2
    #initialize matrix of hough transform values
    out_img = Image.new("L", (ntx, mry), 0)
    pixel_hough_img = out_img.load()

    sqrtd = np.sqrt(im_x**2 +  im_y**2)
    d_r = sqrtd / (mry/2)
    d_theta = np.pi /(2*ntx)
    #iterating through out the image and find pixel with high intensity
    for jx in range(im_x):
        for iy in range(im_y):
            col = clone_im[jx, iy]
            if col == 255: continue
            for jtx in range(ntx):
                th = d_theta * jtx
                r = jx*np.cos(th) + iy*np.sin(th)
                iry = mry/2 + int(r/d_r+0.5)
                pixel_hough_img[jtx, iry] += 1
    return out_img
def inverse_hough(img, hough_matrix):
    """
    :param img: edge image 
    :param hough_matrix: ndarray of hough transformed image
    :param threshold: optional threshold
    :return: image with inversed 
    """
    if isinstance(img, np.ndarray):
        clone_im = img
        im_x, im_y = img.shape
    else:
        clone_im = img.load()
        im_x, im_y = img.size
    ntx, mry = im_x, im_y
    sqrtd = np.sqrt(im_x ** 2 + im_y ** 2)
    mry = np.int(mry/2)*2
    #initialize matrix of hough transform values
    out_img = Image.new("L", (im_x, im_y), 0)
    pixel_out_img = out_img.load()
    d_theta = np.pi / (2*ntx)
    for k in range(hough_matrix.shape[0]):
        for l in range(hough_matrix.shape[1]):
            tmp_1 = 0.0
            if hough_matrix[k][l] == 255: continue
            for i in range(clone_im.shape[0]):
                tmp_2 = np.float(l*2.0*sqrtd/(hough_matrix.shape[1]-1) - sqrtd)
                if np.sin(d_theta*k) == 0: tmp_1+=1
                else:
                    tmp_1 = (tmp_2 - i*np.cos(d_theta*k))/np.sin(d_theta*k) + 0.5
                j = np.int(np.floor(tmp_1))
                if 0<= j < mry and clone_im[i][j] == 255:
                    pixel_out_img[j,i] = 255
    return out_img

from skimage.feature import canny
from matplotlib import cm
def test():
    "Test Hough transform with pentagon."
    from PIL import Image
    from scipy import signal as sg
    import utils
    threshold_upper = 100
    str_image_name = 'Gaussian_img_1'
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    image = Image.open(str_image_name + '.jpg')
    image = np.array(image)
    vertical_edge = sg.convolve2d(image, kernel_x, mode='same')
    horizontal_edge = sg.convolve2d(image, kernel_y, mode='same')
    #binary thresholding
    edge_image = utils.func_verify_image(vertical_edge, threshold_1=threshold_upper)\
                 + utils.func_verify_image(horizontal_edge, threshold_1=threshold_upper)

    edge_image = utils.func_verify_image(edge_image, threshold_1=threshold_upper)

    # edges1 = canny(image,low_threshold=10)
    hough_img = hough(edge_image)
    greyscale_map = list(hough_img.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape(edge_image.shape)
    threshold_hough_image = utils.func_verify_image(greyscale_map, threshold_1= 250)
    inverse_hough_img = inverse_hough(edge_image, threshold_hough_image)
    # Generating figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 5), sharex=True, sharey=True)
    plt.gcf().canvas.set_window_title('Project 2')
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edge_image, cmap=cm.gray)
    ax[1].set_title('Sobel edges')

    ax[2].imshow(hough_img, cmap=cm.gray)
    ax[2].set_title('Hough Transform')

    ax[3].imshow(inverse_hough_img)
    ax[3].set_title('Inverse Transform')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()

    plt.show()
    Image.fromarray(edge_image).save('Edge with noise {0}.jpg'.format(str_image_name))
    hough_img.save('Hough transform image {0}.jpg'.format(str_image_name))
    inverse_hough_img.save('Inverse Hough Transform {0}.jpg'.format(str_image_name))
if __name__ == "__main__": test()
