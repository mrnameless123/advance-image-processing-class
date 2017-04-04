import numpy as np
import warnings
from sklearn.preprocessing import normalize
from enum import Enum
warnings.filterwarnings('error')
print('Imported',__name__)
class BroadCastType(Enum):
    BC_VERTICAL = 0
    BC_HORIZONTAL = 1

def func_compute_rms(param_input):
    print(np.linalg.norm(param_input/255))
    # sumvalue = 0
    # for x in param_input:
    #     for y in x:
    #         sumvalue += y**2
    # print(np.sqrt(sumvalue))


def compute_distance(array_a, array_b):
    distance = []
    for x in array_a:
        tmp = []
        for y in array_b:
            try:
                item = x-y
            except Exception as argument:
                print('Exception Occured:',argument)
                return
            else:
                value = np.power(np.linalg.norm(item),2)
                tmp.append(value)
        distance.append(tmp)
    return np.array(distance)
def index_of_value_in_array(param_array, param_value):
    array = np.array(param_array)
    if array.ndim == 1:
        result1d = np.where(param_array == param_value)
        try:
            result1d[0]
        except Exception as argument:
            print('index_of_value_in_array Exception occurred: {0}'.format(argument))
            return None
        else:
            return np.array(result1d)
    elif array.ndim == 2:
        result2d = np.where(param_array == param_value)
        try:
            result2d[0][0]
        except Exception as argument:
            print('index_of_value_in_array Exception occurred: {0}'.format(argument))
            return None
        else:
            tmp1 = result2d[0]
            tmp2 = result2d[1]
            return  np.vstack([tmp1,tmp2]).T
    elif array.ndim == 3:
        result3d = np.where(param_array == param_value)
        try:
            result3d[0][0][0]
        except Exception as argument:
            print('index_of_value_in_array Exception occurred: {0}'.format(argument))
            return None
        else:
            tmp1 = result3d[0]
            tmp2 = result3d[1]
            tmp3 = result3d[2]
            return np.vstack([tmp1,tmp2,tmp3]).T

def func_broadcast_array(param_broadcaster, param_new_shape, param_broadcast_axis = BroadCastType.BC_VERTICAL):
    clone = np.array(param_broadcaster)
    output = np.array(param_broadcaster)
    iteration = int(param_new_shape)
    if  param_broadcast_axis == BroadCastType.BC_VERTICAL:
        for x in range(iteration - 1):
            output = np.vstack([output, clone])
        if output.shape[0] != iteration:
            print('func_broadcast_array experienced exception: {0} != {1} shape is {2}'.format(int(output.shape[1]), iteration, np.shape(output)))
            input()
        return  output
    elif param_broadcast_axis == BroadCastType.BC_HORIZONTAL:
        for x in range(iteration - 1):
            output = np.hstack([output, clone])
        if output.shape[1] != iteration:
            print('func_broadcast_array experienced exception: {0} != {1} shape is {2}'.format(int(output.shape[0]), iteration, np.shape(output)))
            input()
        return output

def func_my_normalize(param_input):
    row, col = param_input.shape
    tmp = np.reshape(param_input, (1, row * col)).astype(np.float64)
    normed_matrix = normalize(tmp.astype(np.float64), axis=1, norm='max')
    normed_image = np.reshape(normed_matrix, param_input.shape)
    max_value = np.amax(param_input)
    return normed_image, max_value
def func_verify_image(param_input):
    try:
        clone = param_input.copy()
        x, y = clone.shape
        for i in range(x):
            for j in range(y):
                if clone[i][j] > 255:
                    clone[i][j] = 255
                elif clone[i][j] < 0:
                    clone[i][j] = 0
    except Exception as Argument:
        print('func_verify_image exception occurred: {0}'.format(param_input))
        input()
    else:
        return np.array(clone, dtype= np.uint8)

def func_add_noisy(image, noise_typ = 'gaussian', **kwargs):
    mode = noise_typ.lower()
    allowed_types = {
        'gaussian': 'gaussian_values',
        'laplacian': 'laplacian_values',
        'local_var': 'local_var_values',
        'poisson': 'poisson_values',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}
    kw_defaults = {
        'mean': 0.,
        'var': 0.01,
        'exponential_decay' : 1.0,
        'amount': 0.005,
        'salt_vs_pepper': 0.5,
        'local_vars': np.zeros_like(image) + 0.01}

    allowed_kwargs = {
        'gaussian_values': ['mean', 'var'],
        'laplacian_values': ['mean', 'exponential_decay'],
        'local_var_values': ['local_vars'],
        'sp_values': ['amount', 'salt_vs_pepper'],
        's&p_values': ['amount', 'salt_vs_pepper'],
        'poisson_values': []}
    #Check if the parameter is correct or not
    for key in kwargs:
        if key not in allowed_kwargs[allowed_types[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowed_kwargs[allowed_types[mode]]))
    for kw in allowed_kwargs[allowed_types[mode]]:
        kwargs.setdefault(kw, kw_defaults[kw])
    #Normalize input image
    row, col = image.shape
    tmp = np.reshape(image, (1, row*col)).astype(np.float64)
    normed_matrix = normalize(tmp.astype(np.float64), axis=1, norm='max')
    normed_image = np.reshape(normed_matrix, image.shape)
    max_value = np.amax(image)
    #Process add noise to image according to type of noise
    if noise_typ == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 normed_image.shape)
        func_compute_rms(noise)
        noised_img = normed_image + noise
        return func_verify_image(noised_img*max_value)
    elif noise_typ == 'laplacian':
        '''y = ln(2x) if x\in(0, 0.5) otherwise -ln(2-2x) if x\in(0.5, 1)
            f(x; \mu, \lambda) = \frac{1}{2\lambda} \exp\left(-\frac{|x - \mu|}{\lambda}\right).
        '''
        noise = np.random.laplace(kwargs['mean'], kwargs['exponential_decay'], normed_image.shape)
        noised_img = normed_image + noise;
        return func_verify_image(noised_img*max_value)
    elif noise_typ == 'salt':
        out = normed_image
        # Salt mode
        num_salt = np.ceil(kwargs['amount'] * normed_image.size * kwargs['salt_vs_pepper'])
        co_ordinates = [np.random.randint(0, i - 1, int(num_salt))
                  for i in normed_image.shape]
        out[co_ordinates] = 1
        return func_verify_image(out*max_value)
    elif noise_typ == 'pepper':
        # Pepper mode
        out = normed_image
        num_pepper = np.ceil(kwargs['amount'] * normed_image.size * (1. - kwargs['salt_vs_pepper']))
        co_ordinates = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in normed_image.shape]
        out[co_ordinates] = 0
        return func_verify_image(out*max_value)
    elif noise_typ == "s&p":
        out = normed_image
        # Salt mode
        num_salt = np.ceil(kwargs['amount'] * normed_image.size * kwargs['salt_vs_pepper'])
        co_ordinates = [np.random.randint(0, i - 1, int(num_salt))
                        for i in normed_image.shape]
        out[co_ordinates] = 1

        # Pepper mode
        num_pepper = np.ceil(kwargs['amount'] * normed_image.size * (1. - kwargs['salt_vs_pepper']))
        co_ordinates = [np.random.randint(0, i - 1, int(num_pepper))
                        for i in normed_image.shape]
        out[co_ordinates] = 0
        return func_verify_image(out*max_value)
    elif noise_typ == 'poisson':
        values = len(np.unique(normed_image))
        values = 2 ** np.ceil(np.log2(values))
        noisy = np.random.poisson(normed_image * values) / float(values)
        return func_verify_image(noisy*max_value)
    elif noise_typ == 'speckle':
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = normed_image + normed_image * gauss
        return func_verify_image(noisy*max_value)

def func_manual_rotate_image_interpolation(param_input, param_angle, param_interpolation=0):
     rows, cols = np.array(param_input).shape
     ratio = (param_angle / 180) * np.pi
     origin_center = np.array([np.round(rows / 2), np.round(cols / 2)])
     new_row = np.int(np.abs(rows * np.cos(ratio)) + np.abs(cols * np.sin(ratio)))
     new_col = np.int(np.abs(rows * np.sin(ratio)) + np.abs(cols * np.cos(ratio)))
     output = np.zeros((new_row, new_col))
     kernel = np.array([[np.cos(ratio), - np.sin(ratio)], [np.sin(ratio), np.cos(ratio)]])
     output_center = np.array([np.round(new_row / 2), np.round(new_col / 2)])
     if param_interpolation == 0:  # Nearest interpolation
         try:
            for x in range(new_row):
               for y in range(new_col):
                    vector_rotate = np.array([x - output_center[0], y - output_center[1]])
                    tmp = np.dot(kernel.T, vector_rotate)
                    rotate_momentum = np.array(tmp + origin_center, dtype=np.int)
                    if 0 < rotate_momentum[0] < rows and 0 < rotate_momentum[1] < cols:
                        output[x][y] = param_input[rotate_momentum[0]][rotate_momentum[1]]
         except Exception as Argument:
            print('func_manual_rotate_image_interpolation exception occurred: {0}'.format(Argument))
            input()
         else:
            return output
     elif param_interpolation == 1:  # Bilinear interpolation
         try:
             for x in range(new_row):
                 for y in range(new_col):
                     vector_rotate = np.array([x - output_center[0], y - output_center[1]])
                     tmp = np.dot(kernel.T, vector_rotate)
                     rotate_momentum = tmp + origin_center
                     x1, y1 = np.floor(rotate_momentum).astype(dtype=np.int)
                     x2, y2 = np.ceil(rotate_momentum).astype(dtype=np.int)
                     if 0 < x1 < rows and 0 < y1 < cols and 0 < x2 < rows and 0 < y2 < cols:
                         pol1 = param_input[x1][y1]
                         pol2 = param_input[x1][y2]
                         pol3 = param_input[x2][y1]
                         pol4 = param_input[x2][y2]
                         bilinear_pixel_value1 = pol1*(x2 - rotate_momentum[0])*(y2 - rotate_momentum[1]) + pol2*(rotate_momentum[0] - x1)*(y2 - rotate_momentum[1])
                         bilinear_pixel_value2 = pol3*(x2 - rotate_momentum[0])*(rotate_momentum[1] - y1) + pol4*(rotate_momentum[0] - x1)*(rotate_momentum[1] - y1)
                         bilinear_pixel_value = bilinear_pixel_value1 + bilinear_pixel_value2
                         if bilinear_pixel_value < 0:
                             bilinear_pixel_value = 0
                         elif bilinear_pixel_value > 255:
                             bilinear_pixel_value = 1
                         output[x][y] = np.uint8(bilinear_pixel_value)
         except Exception as Argument:
            print('func_manual_rotate_image_interpolation exception occurred: {0}'.format(Argument))
            input()
         else:
            return output

# This function visualizes filters in matrix A. Each column of A is a
# filter. We will reshape each column into a square image and visualizes
# on each cell of the visualization panel.
# All other parameters are optional, usually you do not need to worry
# about it.
# opt_normalize: whether we need to normalize the filter so that all of
# them can have similar contrast. Default value is true.
# opt_graycolor: whether we use gray as the heat map. Default is true.
# opt_colmajor: you can switch convention to row major for A. In that
# case, each row of A is a filter. Default value is false.
# source: https://github.com/tsaith/ufldl_tutorial

def display_network(A, m=-1, n=-1):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    if m < 0 or n < 0:
        n = np.ceil(np.sqrt(col))
        m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0

    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    return image


def display_color_network(A):
    """
    # display receptive field(s) or basis vector(s) for image patches
    #
    # A         the basis, with patches as column vectors
    # In case the midpoint is not set at 0, we shift it dynamically
    :param A:
    :param file:
    :return:
    """
    if np.min(A) >= 0:
        A = A - np.mean(A)

    cols = np.round(np.sqrt(A.shape[1]))

    channel_size = A.shape[0] / 3
    dim = np.sqrt(channel_size)
    dimp = dim + 1
    rows = np.ceil(A.shape[1] / cols)

    B = A[0:channel_size, :]
    C = A[channel_size:2 * channel_size, :]
    D = A[2 * channel_size:3 * channel_size, :]

    B = B / np.max(np.abs(B))
    C = C / np.max(np.abs(C))
    D = D / np.max(np.abs(D))

    # Initialization of the image
    image = np.ones(shape=(dim * rows + rows - 1, dim * cols + cols - 1, 3))

    for i in range(int(rows)):
        for j in range(int(cols)):
            # This sets the patch
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 0] = B[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 1] = C[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 2] = D[:, i * cols + j].reshape(dim, dim)

    image = (image + 1) / 2

    # PIL.Image.fromarray(np.uint8(image * 255), 'RGB').save(filename)

    return image