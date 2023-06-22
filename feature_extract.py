import os

import cv2
import numpy as np
from skimage import exposure, feature
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def GCLM(image): #GCLM特征
    '''
    example:
        values = []
        values.append([])
        temp_ = GCLM(image)
        values[0].append(np.array(temp_).ravel())
    '''
    values_temp = []
    input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(input, [2, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)  # , np.pi / 4, np.pi / 2, np.pi * 3 / 4
    print(glcm.shape) 
    for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = graycoprops(glcm, prop)
        values_temp.append(temp)
    return (values_temp)

def hog(image): # 方向梯度直方图特征
    fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), channel_axis=-1, visualize=True)
    return fd

def LBP(image): #lbp特征（边缘 轮廓）
    input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(input, P=8, R=1)
    return lbp

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(cur_dir, 'data')
    pass
    '''
    
    add your data code
    
    '''

