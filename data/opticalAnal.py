import os
import cv2
import random
import glob
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
VIDEO_TYPE = '.MP4'
FRAMES = 553
FRAMES_H_REDS = 720
FRAMES_W_REDS = 1280
FRAMES_H_REAL = 1080
FRAMES_W_REAL = 1920
THRES = 0.25
STEP = 0.1
MAX = 5


def draw():
    pp = PdfPages('/home/wangzerun/SOTA/LocalBlurNet/final.pdf')
    plot.figure(figsize=(8, 7.2))
    thres_list = []
    reds = np.load('/home/wangzerun/SOTA/LocalBlurNet/reds.npy')
    gopro = np.load('/home/wangzerun/SOTA/LocalBlurNet/gopro.npy')
    lode = np.load('/home/wangzerun/SOTA/LocalBlurNet/lode.npy')
    count = 0
    for t_i in np.arange(0, MAX+STEP, STEP):

        if count % 10 == 0:
            thres_list.append('{:.2f}'.format(t_i))
        else:
            thres_list.append('')
        count += 1
    plot.xlabel('Optical Flow Threshold ', fontsize=25)
    plot.ylabel('Blurred Area Ratio', fontsize=25)
    x = np.arange(len(reds))
    a = plot.bar(x-0.3, reds, 0.3, color='blue', label='REDS', align='center')
    b = plot.bar(x, gopro, 0.3, color='yellow', label='GoPro', align='center')
    c = plot.bar(x+0.3, lode, 0.3, color='red', label='LODE', align='center')
    plot.xticks(x, thres_list, fontsize=20)
    plot.yticks(fontsize=20)
    plot.legend(fontsize=20)
    pp.savefig()
    pp.close()

    #plot.show()

def frameGeneratorGoPro(videoPath='/home/wangzerun/data/gopro/test/'):
    thres_list = []
    for t_i in np.arange(0, MAX+STEP, STEP):
        thres_list.append(0)
    targetFiles = glob.glob(videoPath + '*')
    frameCount = 0
    for f in targetFiles:
        print(f)
        frameCount += 1
        frameNum = 3
        path = f.split('/')[-1]

        FRAMES = glob.glob(f + '/*')
        FRAMES = sorted(FRAMES)
        #FRAMES = (lambda x: (x.split('/')[-1].split('.')[0].sort(), x)[1])(FRAMES)

        frameStart = random.randint(1, len(FRAMES)- 2)

        image_id = FRAMES[frameStart]
        gt_frame = cv2.imread(image_id)
        image_id = FRAMES[frameStart+2]
        nx_gt_frame = cv2.imread(image_id)

        now = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(nx_gt_frame, cv2.COLOR_BGR2GRAY)
        inst = cv2.optflow.createOptFlow_DeepFlow()

        flow = inst.calc(now, next, None)
        flow_map = np.zeros(flow.shape[:2])
        flow_map = np.power(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2), 0.5)
        count = 0
        for t_i in np.arange(0, MAX+STEP, STEP):
            indexs = np.where(flow_map > t_i)
            ratio = len(indexs[0]) / (FRAMES_H_REDS * FRAMES_W_REDS)
            thres_list[count] += ratio
            count += 1
    for i in range(len(thres_list)):
        thres_list[i] = thres_list[i] / frameCount
    thres_list = np.asarray(thres_list)
    np.save('/home/wangzerun/SOTA/LocalBlurNet/gopro.npy', thres_list)
    plot.bar(range(len(thres_list)), thres_list, width=0.4)
    plot.show()


def frameGeneratorREDS(videoPath='/home/wangzerun/data/reds/val/val_orig/'):
    thres_list = []
    for t_i in np.arange(0, MAX+STEP, STEP):
        thres_list.append(0)
    targetFiles = glob.glob(videoPath+'*')
    frameCount = 0
    for f in targetFiles:
        print(f)
        frameCount += 1
        frameNum = 3
        path = f.split('/')[-1]

        FRAMES = len(glob.glob(f+'/*'))
        frameStart = random.randint(1, FRAMES - frameNum)
        image_id = str(frameStart + (frameNum // 2)).rjust(8,'0')
        gt_frame = cv2.imread(f+'/'+image_id+'.png')
        image_id = str(frameStart + (frameNum // 2) - 1).rjust(8, '0')
        pr_gt_frame = cv2.imread(f + '/' +  image_id + '.png')
        image_id = str(frameStart + (frameNum // 2) + 1).rjust(8, '0')
        nx_gt_frame = cv2.imread(f + '/' + image_id + '.png')
        prev = cv2.cvtColor(pr_gt_frame, cv2.COLOR_BGR2GRAY)
        now = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(nx_gt_frame, cv2.COLOR_BGR2GRAY)
        inst = cv2.optflow.createOptFlow_DeepFlow()

        flow = inst.calc(prev, now, None)
        flow_map = np.zeros(flow.shape[:2])
        flow_map = np.power(np.power(flow[:,:,0],2)+np.power(flow[:,:,1],2),0.5)
        count = 0
        for t_i in np.arange(0, MAX+STEP, STEP):
            indexs = np.where(flow_map > t_i)
            ratio = len(indexs[0]) / (FRAMES_H_REDS*FRAMES_W_REDS)
            thres_list[count] += ratio
            count += 1
    for i in range(len(thres_list)):
        thres_list[i] = thres_list[i] / frameCount
    thres_list = np.asarray(thres_list)
    np.save('/home/wangzerun/SOTA/LocalBlurNet/reds.npy', thres_list)
    plot.bar(range(len(thres_list)), thres_list, width=0.4)
    plot.show()


def frameGeneratorReal(videoPath='/home/wangzerun/data/realframe/'):
    thres_list = []
    for t_i in np.arange(0, MAX+STEP, STEP):
        thres_list.append(0)
    targetFiles = glob.glob(videoPath + '*')
    frameCount = 0
    for f in targetFiles:
        print(f)
        frameCount += 1
        frameNum = 2
        path = f.split('/')[-1]

        FRAMES = glob.glob(f + '/*')
        frame1 = FRAMES[0]

        pr_gt_frame = cv2.imread(FRAMES[0])

        nx_gt_frame = cv2.imread(FRAMES[1])
        prev = cv2.cvtColor(pr_gt_frame, cv2.COLOR_BGR2GRAY)

        next = cv2.cvtColor(nx_gt_frame, cv2.COLOR_BGR2GRAY)
        inst = cv2.optflow.createOptFlow_DeepFlow()

        flow = inst.calc(prev, next, None)
        flow_map = np.zeros(flow.shape[:2])
        flow_map = np.power(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2), 0.5)
        count = 0
        for t_i in np.arange(0, MAX+STEP, STEP):
            indexs = np.where(flow_map > t_i)
            ratio = len(indexs[0]) / (FRAMES_H_REAL * FRAMES_W_REAL)
            thres_list[count] += ratio
            count += 1
    for i in range(len(thres_list)):
        thres_list[i] = thres_list[i] / frameCount
    thres_list = np.asarray(thres_list)
    np.save('/home/wangzerun/SOTA/LocalBlurNet/lode.npy', thres_list)
    plot.bar(range(len(thres_list)), thres_list, width=0.4)
    plot.show()

if __name__ == '__main__':
    #frameGeneratorGoPro()
    draw()