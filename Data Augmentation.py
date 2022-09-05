import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
import base64
import json
import re
from copy import deepcopy
import argparse



class DataAugment():
    def __init__(self, 
                 change_light_rate = 0.5,
                 add_noise_rate = 0.5, 
                 random_point = 0.5, 
                 flip_rate = 0.5, 
                 shift_rate = 0.5, 
                 rotate_rate = 0.5,
                 cropped_rate = 0.5,
                 rand_point_percent = 0.03,
                 is_addNoise = True, 
                 is_changeLight = True, 
                 is_random_point = True, 
                 is_shift_img = True,
                 is_filp_img = True,
                 is_rotate = True,
                 is_cropped = True):

        # 配置各个操作的属性
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.random_point = random_point
        self.flip_rate = flip_rate
        self.shift_rate = shift_rate
        self.rotate_rate = rotate_rate
        self.cropped_rate = cropped_rate

        self.rand_point_percent = rand_point_percent

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_random_point = is_random_point
        self.is_filp_img = is_filp_img
        self.is_shift_img = is_shift_img
        self.is_rotate = is_rotate
        self.is_cropped = is_cropped


    # 平移
    def _shift_img(self, img):

        # ---------------------- 平移图像 ----------------------

        h, w, _ = img.shape

        d_to_left = w  # 包含所有目标框的最大左移动距离
        d_to_right = w  # 包含所有目标框的最大右移动距离
        d_to_top = h  # 包含所有目标框的最大上移动距离
        d_to_bottom = h  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 5, (d_to_right - 1) / 5)
        y = random.uniform(-(d_to_top - 1) / 5, (d_to_bottom - 1) / 5)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return shift_img


    #裁剪
    def _cropped(self, img):
        '''
        裁剪
        :param img: 原始图片
        :param x1: 左边界
        :param y1: 上边界
        :param x2: 右边界
        :param y2: 下边界
        :return:
        '''
        h, w, _ = img.shape

        x1 = int(random.uniform(0, (w - 1) / 5))
        y1 = int(random.uniform(0, (h - 1) / 5))
        x2 = int(random.uniform(4 * (w - 1) / 5, w - 1))
        y2 = int(random.uniform(4 * (h - 1) / 5, h - 1))
        
        resultImg = img[y1:y2, x1:x2]
        return resultImg

    #镜像
    def _filp_img(self, img):

        # ---------------------- 翻转图像 ----------------------

        sed = random.random()

        if 0 < sed < 0.33:  # 0.33的概率水平翻转，0.33的概率垂直翻转,0.33是对角反转
            flip_img = cv2.flip(img, 0)  # _flip_x

        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(img, 1)  # _flip_y

        else:
            flip_img = cv2.flip(img, -1)  # flip_x_y

        return flip_img


    # 调整亮度
    def _changeLight(self, img):

        alpha = random.uniform(0.5, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)


    # 旋转
    def _rotate(self, image, center=None, scale=1.0):

        angle = random.uniform(0, 360)
        (h, w) = image.shape[:2]
        # If no rotation center is specified, the center of the image is set as the rotation center
        if center is None:
            center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, m, (w, h))
        return rotated


    # 加噪声
    def _addNoise(self, img):
        return random_noise(img, seed=int(time.time())) * 255


    #加随机点
    def _addRandPoint(self, img):
        percent = self.rand_point_percent
        num = int(percent * img.shape[0] * img.shape[1])
        for i in range(num):
            rand_x = random.randint(0, img.shape[0] - 1)
            rand_y = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[rand_x, rand_y] = 0
            else:
                img[rand_x, rand_y] = 255
        return img


    def dataAugment(self, img):

        change_num = 0  # 改变的次数
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)

            if self.is_random_point:
                if random.random() < self.random_point:  # 加随机点
                    change_num += 1
                    img = self._addRandPoint(img)


            if self.is_shift_img:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img = self._shift_img(img)


            if self.is_filp_img:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img = self._filp_img(img)

            if self.is_rotate:
                if random.random() < self.rotate_rate:  # 旋转
                    change_num += 1
                    img = self._rotate(img)

            if self.is_cropped:
                if random.random() < self.cropped_rate:  # 裁剪
                    change_num += 1
                    img = self._cropped(img)

        return img



dataAug = DataAugment()
im =cv2.imread("0001.jpg")
res = dataAug.dataAugment(im)
cv2.imwrite("res.jpg", res)