import os
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


REDS_ROOT = '/dataset/REDS_yf/train/sharp'
VOC_ROOT = '/home/wangzerun/data/VOCdevkit/VOC2012/'
KERNEL_ROOT = '/home/wangzerun/BlurGenerator/DeblurGanBlur/psf'

def extractLabelListREDS(data_path):
    label_list = glob.glob(data_path+'/*.png')
    return label_list

def extractLableListVOC(data_path):
    label_prefix = 'SegmentationObject'
    label_path = os.path.join(data_path, label_prefix)
    label_list = glob.glob(label_path+'/*.png')
    return label_list

def extractLabelListKernel(data_path, select_min, select_max):
    label_list = glob.glob(data_path+'/*.npy')
    kernel_list = []
    for l in label_list:
        size = int(l.split('/')[-1].split('_')[1].split('.')[0])
        if size >= select_min and size < select_max:
            kernel_list.append(l)
    return kernel_list


def label2imgVOC(label_id):
    label_id = label_id.replace("SegmentationObject","JPEGImages")
    return label_id[:-3]+'jpg'

def img2label(img_id):
    return img_id[:-3] + 'png'

def img2labelVOC(img_id):
    img_id = img_id.replace("JPEGImages", "SegmentationObject")
    return img_id[:-3] + 'png'


def extractPngLabel(label_id):
    label = cv2.imread(label_id,cv2.IMREAD_GRAYSCALE)
    #label = cv2.resize(label,None,fx=1.5,fy=1.5)
    return label


def motion_blur(image,blur_kernel):
    image = np.array(image)
    blurred = cv2.filter2D(image, -1, blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


def motion_blur_kernel(degree, angle):
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    return kernel

def normalvariate_random_int(mean, variance, dmin, dmax):
    r = dmax + 1
    while r < dmin or r > dmax:
        r = int(random.normalvariate(mean, variance))
    return r


def uniform_random_int(dmin, dmax):
    r = random.randint(dmin,dmax)
    return r


def random_blur_kernel(mean=50, variance=15, dmin=10, dmax=100):
    random_degree = normalvariate_random_int(mean, variance, dmin, dmax)
    random_angle = uniform_random_int(-180, 180)
    return motion_blur_kernel(random_degree,random_angle)


if __name__ == '__main__':
    kernel = random_blur_kernel(mean=50, variance=15, dmin=10, dmax=100)

    plt.imshow(kernel)
    plt.show()