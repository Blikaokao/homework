import numpy as np
#生成一个掩码 就是高斯分布
def gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    #出现0 替换成很小的eps
    h[h<np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        #归一化
        h/= sumh
    return h