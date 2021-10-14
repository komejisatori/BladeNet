import os
import sys
sys.path.append('../')
import cv2
import random
import math
import numpy as np
from Generator.NewBlurGenerator.util import *
import matplotlib.pyplot as plt

def putInObject(object, mask, background, position):
    pos_h = position[0]
    pos_w = position[1]
    ob_h = object.shape[0]
    ob_w = object.shape[1]
    bg_h = background.shape[0]
    bg_w = background.shape[1]
    y1 = pos_h
    x1 = pos_w
    y2 = pos_h + ob_h
    x2 = pos_w + ob_w
    channel = len(background.shape)
    if (x1 < bg_w) and (y1 < bg_h) and (x2 > 0) and (y2 > 0): # check
        ob_y1 = 0
        ob_x1 = 0
        bg_y1 = y1
        bg_x1 = x1
        ob_y2 = ob_h
        ob_x2 = ob_w
        bg_y2 = y2
        bg_x2 = x2

        if(bg_y1 < 0):
            bg_y1 = 0
            ob_y1 = - pos_h
        if(bg_x1 < 0):
            bg_x1 = 0
            ob_x1 = - pos_w
        if(bg_y2 >= bg_h):
            bg_y2 = bg_h
            ob_y2 = bg_h - pos_h
        if(bg_x2 >= bg_w):
            bg_x2 = bg_w
            ob_x2 = bg_w - pos_w

        alpha_ob = mask[ob_y1:ob_y2, ob_x1:ob_x2]
        alpha_bg = 1.0 - alpha_ob
        if channel < 3:
            background[bg_y1:bg_y2, bg_x1:bg_x2] = alpha_ob * object[ob_y1:ob_y2, ob_x1:ob_x2] + \
                                                      alpha_bg * background[bg_y1:bg_y2, bg_x1:bg_x2]
        else:
            for c in range(0, 3):
                background[bg_y1:bg_y2,bg_x1:bg_x2,c] = alpha_ob * object[ob_y1:ob_y2, ob_x1:ob_x2, c] + \
                                                    alpha_bg * background[bg_y1:bg_y2, bg_x1:bg_x2, c]


    return background


def putInObjects(front_objects, front_objects_masks, front_objects_kernels, front_objects_kernel_sizes, static_objects,static_objects_masks, background):
    global_blur = False
    blur_bg = background.copy()
    sharp_bg = background.copy()
    mask = np.zeros(background.shape[:2])
    bg_h, bg_w = background.shape[0], background.shape[1]
    for i, o in enumerate(static_objects):
        initial_pos = [random.randint(0, bg_h - 1) - o.shape[0] // 2,
                       random.randint(0, bg_w - 1) - o.shape[1] // 2]

        blur_bg = putInObject(o, static_objects_masks[i], blur_bg, initial_pos)
        sharp_bg = putInObject(o, static_objects_masks[i], sharp_bg, initial_pos)

    for i,o in enumerate(front_objects):
        if np.all(front_objects_masks[i]==0):
            global_blur = True
            print('make global blur')
        kernel = front_objects_kernels[i]
        kernel = kernel / np.sum(kernel)
        motion_blur_img = motion_blur(o, kernel)
        motion_blur_mask = motion_blur(front_objects_masks[i], kernel)
        motion_blur_mask = motion_blur_mask / 255.0
        size = front_objects_kernel_sizes[i]
        gt_mask = np.where(motion_blur_mask > 0, 1, 0)

        initial_pos = [random.randint(0, bg_h - 1) - o.shape[0] // 2,
                       random.randint(0, bg_w - 1) - o.shape[1] // 2]

        blur_bg = putInObject(motion_blur_img, motion_blur_mask, blur_bg, initial_pos)
        sharp_bg = putInObject(o, front_objects_masks[i], sharp_bg, initial_pos)
        mask = putInObject(gt_mask*size, gt_mask, mask, initial_pos)
    #global blur


    return blur_bg, sharp_bg, mask


def dePadImage(img, m, extraPadding=0):

    temp_img = img.copy().astype(np.float32)
    mask = temp_img[:,:,0] + temp_img[:,:,1] + temp_img[:,:,2]

    idx = np.argwhere(np.all(mask[..., :] == 0, axis=0))
    img = np.delete(img, idx, axis=1)
    idx = np.argwhere(np.all(mask[:, ...] == 0, axis=1))
    img = np.delete(img, idx, axis=0)

    temp_img = m.copy().astype(np.float32)
    mask = temp_img

    idx = np.argwhere(np.all(mask[..., :] == 0, axis=0))
    m = np.delete(m, idx, axis=1)
    idx = np.argwhere(np.all(mask[:, ...] == 0, axis=1))
    m = np.delete(m, idx, axis=0)
    '''
    if img.shape[0] > (720 * 0.45) and img.shape[1] > (1080 * 0.45):

        fx = (1080 * 0.45) / img.shape[1]
        fy = (720 * 0.45) / img.shape[0]
        #if fx > 5 or fy > 5:
        #    return np.zeros(img.shape), np.zeros(m.shape)
        if fx > fy:
            fx = fy
        else:
            fy = fx
        x = math.floor(img.shape[1] * fx)
        y = math.floor(img.shape[0] * fy)
        img = cv2.resize(img, (x, y))
        m = cv2.resize(m, (x, y))

    assert img.shape[0] <= (720 * 0.45) or img.shape[1] <= (1080 * 0.45)
    '''
    '''
    if img.shape[0] < (720 * 0.8) or img.shape[1] < (1080 * 0.8):

        fx = (1080 * 0.8) / img.shape[1]
        fy = (720 * 0.8) / img.shape[0]
        if fx > 5 or fy > 5:
            return np.zeros(img.shape), np.zeros(m.shape)
        if fx > fy:
            fy = fx
        else:
            fx = fy
        x = math.ceil(img.shape[1] * fx)
        y = math.ceil(img.shape[0] * fy)
        img = cv2.resize(img, (x, y))
        m = cv2.resize(m, (x, y))

    assert img.shape[0] >= (720 * 0.8) and img.shape[1] >= (1080 * 0.8)
    '''

    img = np.pad(img, ((extraPadding, extraPadding), (extraPadding, extraPadding), (0, 0)))
    m = np.pad(m, ((extraPadding, extraPadding), (extraPadding, extraPadding)))



    return img, m


def getFrontObjects(front_img_path, front_object_limit=None):
    front_img_path = img2labelVOC(front_img_path)
    label_mask = extractPngLabel(front_img_path)
    ids = np.unique(label_mask).tolist()
    front_ids = []
    for i in ids:
        if i != 0 and i != 220:
            front_ids.append(i)
    id_number = len(front_ids)

    assert id_number > 0
    id_number = random.randint(1, id_number)
    if front_object_limit != None:
        id_number = min(id_number, front_object_limit)
    random.shuffle(front_ids)
    masks = []
    for i in front_ids[:id_number]:
        mask = np.zeros(label_mask.shape)
        index = np.where(label_mask == i)
        mask[index] = 1
        masks.append(mask)
    return masks


def getFrontImgBasedOnBackground(static_img_paths, front_img_paths, background_path, kernel_ids, front_object_limit=None):
    front_objects = []
    front_objects_masks = []
    front_objects_kernels = []
    front_objects_kernel_sizes = []
    static_objects = []
    static_objects_masks = []
    bg_img = cv2.imread(background_path)
    if bg_img.shape[2] == 1:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)

    count = 0
    for front_img_path in static_img_paths:
        front_img = cv2.imread(front_img_path)
        if front_img.shape[2] == 1:
            front_img = cv2.cvtColor(front_img, cv2.COLOR_GRAY2BGR)


        front_masks = getFrontObjects(front_img_path, front_object_limit)

        for m in front_masks:
            count += 1
            front_i = front_img.copy()
            for c in range(3):
                front_i[:,:,c] = front_i[:,:,c] * m
            front_i, m = dePadImage(front_i,m)

            static_objects.append(front_i)
            static_objects_masks.append(m)


    count = 0
    for front_img_path in front_img_paths:
        front_img = cv2.imread(front_img_path)
        if front_img.shape[2] == 1:
            front_img = cv2.cvtColor(front_img, cv2.COLOR_GRAY2BGR)


        front_masks = getFrontObjects(front_img_path, front_object_limit)

        for m in front_masks:
            count += 1
            front_i = front_img.copy()
            kernel_id = random.choice(kernel_ids)
            size = int(kernel_id.split('/')[-1].split('_')[1].split('.')[0])
            kernel = np.load(kernel_id)
            for c in range(3):
                front_i[:,:,c] = front_i[:,:,c] * m
            front_i, front_i_m = dePadImage(front_i, m, extraPadding=10)
            front_objects_kernel_sizes.append(size)
            #cv2.imwrite('move_{}.png'.format(count), front_i)
            front_objects.append(front_i)
            front_objects_masks.append(front_i_m)
            front_objects_kernels.append(kernel)

    blur, sharp, map = putInObjects(front_objects, front_objects_masks, front_objects_kernels, front_objects_kernel_sizes, static_objects,static_objects_masks, bg_img)
    return blur, sharp, map


def main():
    bg_test_ids = extractLabelListREDS(REDS_ROOT)[2000:3000]
    f_test_ids = extractLableListVOC(VOC_ROOT)
    kernel_ids = extractLabelListKernel(KERNEL_ROOT, select_min=0, select_max=40)
    for bg_id in bg_test_ids:
        f_num = random.randint(16,20)
        front_id = random.sample(f_test_ids, f_num)
        front_id = [label2imgVOC(x) for x in front_id]
        static_id = random.sample(f_test_ids, 1)
        static_id = [label2imgVOC(x) for x in static_id]
        foldername = bg_id.split('/')[-1].split('.')[0]
        blur, sharp, map = getFrontImgBasedOnBackground(static_id, front_id, bg_id, kernel_ids, front_object_limit=None)
        cv2.imwrite('/dataset/REDS_AUG/sharp_image_val/{}.png'.format(foldername), sharp, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite('/dataset/REDS_AUG/blur_image_val/{}.png'.format(foldername), blur, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        np.save('/dataset/REDS_AUG/blur_map_val/{}.npy'.format(foldername), map)
        #plt.imshow(map)
        #plt.show()
        img_test = cv2.imread('/dataset/REDS_AUG/sharp_image_val/{}.png'.format(foldername))
        assert (img_test - sharp).any() == False
        print('[Generating] {}'.format(foldername))

if __name__ == '__main__':
    main()
