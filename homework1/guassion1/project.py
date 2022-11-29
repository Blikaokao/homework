import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from guassion1 import progaussion, guassionFilter


 #归一化
def normalize(img):
    img_min = img.min()
    img_max = img.max()
    return (img-img_min)/(img_max - img_min)

#读取图像 转换为浮点数
def setup_image(img1_name,img2_name):
    #main_path = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)
    img1 = img1.astype(np.single)/255
    img2 = img2.astype(np.single)/255
    return img1,img2

#进行滤波
def gen_hybird_image(img1,img2,cutoff_frequency,shape,size=3):
    print("进行滤波")
    #cutoff_frequency
    #img1 = cv2.imread(img1)
    #img2 = cv2.imread(img2)

    gaussion_filter = progaussion.gauss2D(shape=shape, sigma=cutoff_frequency)
    print(gaussion_filter)
    low_fre = guassionFilter.my_filter(img1,gaussion_filter,size)
    print("low_fre",low_fre.shape)
    #cv2.imshow("low_fre",normalize(low_fre))
    imgres = cv2.imread(img2)

    #按照第一张图片的大小进行缩放
    imgres = cv2.resize(imgres, (int(imgres.shape[1] / size), int(imgres.shape[0] / size)))
    print("imgres", imgres.shape)
    high_fre = imgres - guassionFilter.my_filter(img2,gaussion_filter,size)
    print("high_fre",high_fre.shape)
    min_w = low_fre.shape[0] if high_fre.shape[0] > low_fre.shape[0] else high_fre.shape[0]
    min_h = low_fre.shape[1] if high_fre.shape[1] > low_fre.shape[1] else high_fre.shape[1]
    hybird_image = np.add(cv2.resize(low_fre,(min_h,min_w)), cv2.resize(high_fre,(min_h,min_w)))
    print("hybird_image",hybird_image.shape)
    #cv2.imshow("low_fre",normalize(low_fre))
    #cv2.imshow("high_fre",normalize(high_fre))
    #cv2.imshow("hybird_image",normalize(hybird_image))

    #cv2.waitKey(0)
    return  low_fre,high_fre,hybird_image

#for sobel_filter demo
def diff_filter(img1,img2,diff_filter,size=3):

    low_fre = guassionFilter.my_filter(img1, diff_filter, size)
    print("low_fre", low_fre.shape)
    # cv2.imshow("low_fre",normalize(low_fre))
    imgres = cv2.imread(img2)

    # 按照第一张图片的大小进行缩放
    imgres = cv2.resize(imgres, (int(imgres.shape[1] / size), int(imgres.shape[0] / size)))
    print("imgres", imgres.shape)
    high_fre = imgres - guassionFilter.my_filter(img2, diff_filter, size)
    print("high_fre", high_fre.shape)
    min_w = low_fre.shape[0] if high_fre.shape[0] > low_fre.shape[0] else high_fre.shape[0]
    min_h = low_fre.shape[1] if high_fre.shape[1] > low_fre.shape[1] else high_fre.shape[1]
    hybird_image = np.add(cv2.resize(low_fre, (min_h, min_w)), cv2.resize(high_fre, (min_h, min_w)))
    print("hybird_image", hybird_image.shape)
    # cv2.imshow("low_fre",normalize(low_fre))
    # cv2.imshow("high_fre",normalize(high_fre))
    # cv2.imshow("hybird_image",normalize(hybird_image))

    #cv2.waitKey(0)
    return low_fre, high_fre, hybird_image

def save_image(low_fre,high_fre,hybird_image,img1_name,img2_name,res_name):

    plt.imsave(img1_name,normalize(low_fre))
    plt.imsave(img2_name,normalize(high_fre+0.5))
    plt.imsave(res_name,normalize(hybird_image))






