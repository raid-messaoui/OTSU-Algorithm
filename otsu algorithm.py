# this code is for beginners

import cv2 as cv
import numpy as np
from time import perf_counter


def otsu_threshold(img):
    img_h, img_w = img.shape
    histogram = np.zeros(256)

    # calculate the histogram
    for i in range(0, img_h):
        for j in range(0, img_w):
            histogram[img[i, j]] += 1

    total_pixels = img_h * img_w
    sum_pixels = sum([i*hist for i, hist in enumerate(histogram)])
    sum_b = 0
    w_b = w_f = 0
    var_max = 0
    threshold = 0

    for t in range(256):
        w_b += histogram[t]  # Weight Background
        if w_b == 0:
            continue

        w_f = total_pixels - w_b  # Weight Foreground

        if w_f == 0:
            break

        sum_b += t * histogram[t]
        mB = sum_b / w_b  # Mean Background
        mF = (sum_pixels - sum_b) / w_f  # Mean Foreground

        # Calculate Between Class Variance
        varBetween = w_b * w_f * (mB - mF) * (mB - mF);

        # Check if new maximum found
        if varBetween > var_max:
            var_max = varBetween
            threshold = t

    return threshold


def gray2bw(img, t):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 0 if img[i, j] < t else 255
    return img


input_img = cv.imread('3.jpg')
input_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
t0 = perf_counter()
threshold = otsu_threshold(input_img)
print(f"threshold = {threshold}\nexecute time = {perf_counter()-t0}" )
t0 = perf_counter()
img_Out = gray2bw(input_img, threshold)
print(f"filtering time = {perf_counter()-t0}")
cv.imshow('output', img_Out)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('3_bw.jpg', img_Out)

'''print(otsu_threshold(input_img))'''
