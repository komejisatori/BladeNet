import os
import glob
import cv2
import matplotlib.pyplot as plt

COCO_ROOT = '/home/wangzerun/data/new_coco'
REDS_ROOT = '/dataset/REDS_yf/train/sharp'
VOC_ROOT = '/home/wangzerun/data/VOCdevkit/VOC2012/'

def extractLabelList(data_path, data_type='train2017'):
    img_path = os.path.join(data_path, data_type)
    label_list = glob.glob(img_path+'/*.png')
    return label_list

def extractLabelListREDS(data_path):
    label_list = glob.glob(data_path+'/*.png')
    return label_list

def extractLableListVOC(data_path):
    label_prefix = 'SegmentationObject'
    label_path = os.path.join(data_path, label_prefix)
    label_list = glob.glob(label_path+'/*.png')
    return label_list

def extractPngLabel(label_id):
    label = cv2.imread(label_id,cv2.IMREAD_GRAYSCALE)
    #label = cv2.resize(label,None,fx=1.5,fy=1.5)
    return label


def label2img(label_id):
    return label_id[:-3]+'jpg'

def label2imgVOC(label_id):
    label_id = label_id.replace("SegmentationObject","JPEGImages")
    return label_id[:-3]+'jpg'

def img2label(img_id):
    return img_id[:-3] + 'png'

def img2labelVOC(img_id):
    img_id = img_id.replace("JPEGImages", "SegmentationObject")
    return img_id[:-3] + 'png'

def outputDemo(label_id, demo_path='./'):
    img_path = label2img(label_id)
    img = cv2.imread(img_path)
    label = extractPngLabel(label_id)
    plt.matshow(label)
    plt.savefig(demo_path + 'label.jpg')
    cv2.imwrite(demo_path+'img.jpg',img)

if __name__ == '__main__':
    test_id = extractLabelList(COCO_ROOT)[27]
    extractPngLabel(test_id)
    outputDemo(test_id)
