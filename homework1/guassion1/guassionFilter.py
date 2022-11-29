import numpy as np
import cv2
#滤波函数
def my_filter(img, filter,size=3):
    print("进行滤波")
    print(img)
    img = cv2.imread(img, cv2.COLOR_BGR2RGB)
    #img = img[:, :, ::-1]
    img = cv2.resize(img,(int(img.shape[1]/size),int(img.shape[0]/size)))

    #cv2.imshow("img",img)
    im_dim = img.shape
    flt_dim = filter.shape
    img_dim1 = im_dim[0]   #输入图像的长宽
    img_dim2 = im_dim[1]
    flt_dim1 = flt_dim[0]  #滤波器的宽高
    flt_dim2 = flt_dim[1]

    print(img_dim1,img_dim2,flt_dim1,flt_dim2)
    #padding填充  长宽分别填充  避免滤波器不是n * n的形式
    pad_dim1 = int((flt_dim1 + 1)/2)
    pad_dim2 = int((flt_dim2 + 1)/2)
    #图像填充给pad_mat
    pad_mat = np.zeros((img_dim1+2*pad_dim1,img_dim2+2*pad_dim2,3))
    filtered_image = np.zeros((img_dim1,img_dim2,3))
    pad_mat[pad_dim1:img_dim1+pad_dim1,pad_dim2:img_dim2+pad_dim2] = img
    print("img",img)
    print("pad_mat",pad_mat)
    #cv2.imshow("img",img)
    #cv2.imshow("pad_mat",pad_mat)
    print("开始卷积")

    #进行卷积操作
    for d in range(len(img[0][0])): #C
        for i in range(len(img)): #W
            for j in range(len(img[0])): #H
                filtered_image[i][j][d] = sum(sum(np.multiply(filter,pad_mat[i:i+flt_dim1,j:j+flt_dim2,d])))


    #cv2.imshow("filtered_image",filtered_image)
    #cv2.waitKey(0)
    #filtered_image = filtered_image[:, :, ::-1]
    return filtered_image

