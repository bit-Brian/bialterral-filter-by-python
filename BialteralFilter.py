import cv2 as cv
import numpy as np
import math
import time


def bilateral_filter_image(image_matrix, window_length=7, sigma_color=12.5, sigma_space=25
                           , mask_image_matrix=None):
    """

    :param image_matrix: 输入图像
    :param window_length: 滤波核大小(滤波期间使用的每个像素邻域的直径)，高= 宽= window_length，滤波核半径 r= (window_length - 1)/2，滤波核越大，输出图像越模糊。默认值为window_length=7
    :param sigma_color:该参数的值越大，意味着像素邻域(参见sigmspace)中更远的颜色将被混合在一起，从而产生更大的半等色区域。默认值为sigma_color=12.5
    :param sigma_space:空间域sigma，参数值越大，表示距离较远的像素将相互影响，只要它们的颜色足够接近(参见sigmaColor)。当d>0时，它指定邻域大小，与sigma_space无关。否则，d与sigma_space成比例。默认值为sigma_space=25
    :param mask_image_matrix:图像处理区域，默认值为全局
    :return: filter_image_matrix，输出处理后的图像
    """
    mask_image_matrix = np.zeros((image_matrix.shape[0], image_matrix.shape[1])
                                 ) if mask_image_matrix is None else mask_image_matrix  # default: filtering the entire image
    image_matrix = image_matrix.astype(np.int32)  # transfer the image_matrix to type int32，for uint cann't represent the negative number afterward
    channel = 3

    if len(image_matrix.shape) < channel:
        channel = 1

    def limit(x):
        x = 0 if x < 0 else x
        x = 255 if x > 255 else x
        return x

    limit_ufun = np.vectorize(limit, otypes=[np.uint8])

    def look_for_gaussion_table(delta):
        return delta_gaussion_dict[delta]

    def generate_bilateral_filter_distance_matrix(window_length, sigma, chan):
        if chan == 3:
            distance_matrix = np.zeros((window_length, window_length, 3))
        if chan == 1:
            distance_matrix = np.zeros((window_length, window_length))
        left_bias = int(math.floor(-(window_length - 1) / 2))
        right_bias = int(math.floor((window_length - 1) / 2))
        for i in range(left_bias, right_bias + 1):
            for j in range(left_bias, right_bias + 1):
                distance_matrix[i - left_bias][j - left_bias] = math.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))
        return distance_matrix

    delta_gaussion_dict = {i: math.exp(-i ** 2 / (2 * (sigma_color ** 2))) for i in range(256)}
    # to accelerate the process of get the gaussion matrix about color.key:color difference，value:gaussion weight
    look_for_gaussion_table_ufun = np.vectorize(look_for_gaussion_table, otypes=[np.float64])
    # get the gaussion weight about distance directly
    bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(window_length
                                                                                 , sigma_space, channel)

    margin = int(window_length / 2)
    left_bias = math.floor(-(window_length - 1) / 2)
    right_bias = math.floor((window_length - 1) / 2)
    filter_image_matrix = image_matrix.astype(np.float64)

    for i in range(0 + margin, image_matrix.shape[0] - margin):
        for j in range(0 + margin, image_matrix.shape[1] - margin):
            if mask_image_matrix[i][j] == 0:
                filter_input = image_matrix[i + left_bias:i + right_bias + 1,
                               j + left_bias:j + right_bias + 1]  # get the input window
                bilateral_filter_value_matrix = look_for_gaussion_table_ufun(
                    np.abs(filter_input - image_matrix[i][j]))  # get the gaussion weight about color
                bilateral_filter_matrix = np.multiply(bilateral_filter_value_matrix,
                                                      bilateral_filter_distance_matrix)  # multiply color gaussion weight  by distane gaussion weight to get the no-norm weigth matrix
                bilateral_filter_matrix = bilateral_filter_matrix / np.sum(bilateral_filter_matrix
                                                                           , keepdims=False, axis=(0, 1))  # normalize the weigth matrix
                filter_output = np.sum(np.multiply(bilateral_filter_matrix, filter_input), axis=(0, 1))  # multiply the input window by the weigth matrix，then get the sum of channels seperately
                filter_image_matrix[i][j] = filter_output
    filter_image_matrix = limit_ufun(filter_image_matrix)  # limit the range

    return filter_image_matrix

if __name__ == '__main__':
    image_matrix = cv.imread("queban.jpeg")
    # image_matrix = cv.cvtColor(image_matrix, cv.COLOR_BGR2GRAY)
    start_time = time.time()
    bilateral_filtered = bilateral_filter_image(image_matrix, 7, 12.5, 25)
    print("My BilateralFilteredImage spends ", time.time()-start_time, 'seconds!')
    start_time = time.time()
    cv_bilateral_filtered = cv.bilateralFilter(image_matrix, 7, 12.5, 25)
    print("cv2 BilateralFilteredImage spends ", time.time() - start_time, 'seconds!')
    cv.imshow('original', image_matrix)
    cv.imwrite('MyBilateralFilteredImage.png', bilateral_filtered)
    cv.imshow('BilateralFilteredImage', bilateral_filtered)
    cv.imwrite('CvBilateralFilterImage.png', cv_bilateral_filtered)
    cv.imshow('cv_bilateral_filtered', cv_bilateral_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()