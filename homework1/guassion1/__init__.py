from guassion1 import project
from guassion1 import guassionFilter
import matplotlib.pyplot as plt
import cv2
import numpy as np
def tmp():
    img1 = "path"
    img2 = "path"
    img1_last = "name"
    img2_last = "name"
    hybird_last = "name"
    '''img1 = cv2.imread(img1)
    x = img1.shape[0]
    y = img1.shape[1]
    img1 = cv2.resize(img1,(int(y/3),int(x/3)))
    cv2.imshow("img1", img1)
    cv2.waitKey(0)'''
    #传进来的是有后缀的
    low_fre_gua,high_fre_gua,hybird_img_gua = project.gen_hybird_image(img1,img2,30,(15,15))
    #sobel_filter
    low_fre_sobel, high_fre_sobel, hybird_img_sobel = project.diff_filter(img1, img2, np.array([[-1,0,1],[-2,0,-2],[-1,0,1]]))
    #laplacian_filter
    low_fre_lap, high_fre_lap, hybird_img_lap = project.diff_filter(img1, img2, np.array([[0, 1, 0], [1, 5, 1], [0, 1, 0]]))
    #存储的时候要没有后缀的名字
    '''img1 = "image2"
    img2 = "image3"
    tmp = project.gaussion2D.gauss2D(shape=(25,25),sigma=0.9)
    img = guassionFilter.my_filter(img1,tmp)
    plt.figure("gauss filter")
    plt.imshow(project.normalize(img))
    plt.imsave("path",project.normalize(img + 0.5))'''

    img1 = "path"+img1_last
    img2 = "path"+img2_last
    low_fre = low_fre_gua[:, :, ::-1]
    high_fre = high_fre_gua[:, :, ::-1]
    hybird_img = hybird_img_gua[:, :, ::-1]
    project.save_image(low_fre,high_fre,hybird_img,img1,img2,"path"+hybird_last)
    img1 = "path"+img1_last
    img2 = "path"+img2_last
    low_fre = low_fre_sobel[:, :, ::-1]
    high_fre = high_fre_sobel[:, :, ::-1]
    hybird_img = hybird_img_sobel[:, :, ::-1]
    project.save_image(low_fre, high_fre, hybird_img, img1, img2,"path"+hybird_last)
    img1 = "path"+img1_last
    img2 = "path"+img2_last
    low_fre = low_fre_lap[:, :, ::-1]
    high_fre = high_fre_lap[:, :, ::-1]
    hybird_img = hybird_img_lap[:, :, ::-1]
    project.save_image(low_fre, high_fre, hybird_img, img1, img2,"path"+hybird_last)


def tmp1():
    img_my = "path"

    for i  in range(3,20,2):
        j = (i-2)/2
        #j = 0.2 * i + 0.1
        tmp_gaussion_filter = project.progaussion.gauss2D((i, i),3)
        img = guassionFilter.my_filter(img_my, tmp_gaussion_filter)
        img = img[:,:,::-1]
        plt.imsave("path"+str(i) +".png", project.normalize(img + 0.5))
        plt.subplot(331+ j)
        plt.axis('off')
        plt.title('gaussion_k='+str(i))
        plt.imshow(project.normalize(img + 0.5))

    plt.show()

if __name__ == '__main__':
    imgoA = cv2.imread("path")
    imgoA = imgoA[:, :, ::-1]
    imgoB = cv2.imread("path")
    imgoB = imgoB[:, :, ::-1]
    imgA = cv2.imread("path")
    imgA = imgA[:, :, ::-1]
    imgB = cv2.imread("path")
    imgB = imgB[:, :, ::-1]
    imgH = cv2.imread("path")
    imgH = imgH[:, :, ::-1]
    plt.subplot(331 + 0)
    plt.axis('off')
    plt.title('image-A')
    plt.imshow(imgoA)
    plt.subplot(331 + 1)
    plt.axis('off')
    plt.title('image-B')
    plt.imshow(imgoB)
    plt.subplot(331 + 3)
    plt.axis('off')
    plt.title('Lap-high')
    plt.imshow(imgA)
    plt.subplot(331 + 4)
    plt.axis('off')
    plt.title('Lap-low')
    plt.imshow(imgB)
    plt.subplot(331 + 5)
    plt.axis('off')
    plt.title('Hybrid-Image')
    plt.imshow(imgH)
    plt.show()


