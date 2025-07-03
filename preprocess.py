import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess(image_path,thres=110,cover=500,color=(135,135,135),fill_mode="fix"):
    input_img_file=image_path
    img=plt.imread(input_img_file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    mask = img.copy()
    mask=cv2.medianBlur(mask,3)

    '''mean=img.sum(axis=1) #备用，可计算出平均值，用来反转图像
    mean=mean.sum(axis=0)/(img.shape[0]*img.shape[1])'''
    mask=[135.0,135.0,135.0]-mask
    
    
    return np.float32(mask)
if __name__=="__main__":
    input_img_file="F:\\jingxiang\\jingxiang_0714\\2h\\2h\\image0005.tif"
    mask=preprocess(input_img_file,fill_mode="adaptive")
    cv2.namedWindow("mask",0)
    cv2.imshow("mask", mask)
    cv2.waitKey(200)
    cv2.waitKey(0)
