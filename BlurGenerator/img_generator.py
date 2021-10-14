import os
import sys
sys.path.append('../')
import cv2
import random
import numpy as np
from Generator.new_coco_generator import img2label, label2img, extractPngLabel, extractLabelList, COCO_ROOT
from Generator.label_config import FRONT_IDS
from Generator.move_config import FRAME_NUM_LIST, SCALE, ROTATE_ANGLE
import matplotlib.pyplot as plt

def getFrontObjects(front_img_path, front_object_limit=None):
    front_img_path = img2label(front_img_path)
    label_mask = extractPngLabel(front_img_path)
    ids = np.unique(label_mask).tolist()
    front_ids = []
    for i in ids:
        if (i+1) in FRONT_IDS:
            front_ids.append(i)
    id_number = len(front_ids)
    if front_object_limit != None:
        id_number = min(id_number, front_object_limit)
    id_number = random.randint(0, id_number)
    random.shuffle(front_ids)
    masks = []
    for i in front_ids[:id_number]:
        mask = np.zeros(label_mask.shape)
        index = np.where(label_mask == i)
        mask[index] = 1
        masks.append(mask)
    return masks

def dePadImage(img):
    temp_img = img.copy().astype(np.float32)
    mask = temp_img[:,:,0] + temp_img[:,:,1] + temp_img[:,:,2]
    idx = np.argwhere(np.all(mask[..., :] == 0, axis=0))
    img = np.delete(img, idx, axis=1)
    idx = np.argwhere(np.all(mask[:, ...] == 0, axis=1))
    img = np.delete(img, idx, axis=0)
    return img



def generateMotionPath(front_objects, bg_h, bg_w):
    front_object_number = len(front_objects)
    frame_num = random.choice(FRAME_NUM_LIST)

    xy_motions = []
    z_motions = [] # -1 for scale down 0 for nothing 1 for scale up
    r_motions = []  # -1 for left 0 for nothing 1 for right
    initials = []

    for o in range(front_object_number):
        xy_motion = []
        for f in range(frame_num):
            x_step = random.randint(0,1)
            y_step = random.randint(0,1)
            xy_motion.append([y_step, x_step])
        xy_motions.append(xy_motion)

    for o in range(front_object_number):
        z_motion = random.randint(-1, 1)  # -1 for scale down 0 for nothing 1 for scale up
        r_motion = random.randint(-1, 1)  # -1 for left 0 for nothing 1 for right
        z_motions.append(z_motion)
        r_motions.append(r_motion)

        initial_pos = [random.randint(0, bg_h-1)-front_objects[o].shape[0] // 2,
                       random.randint(0, bg_w-1)-front_objects[o].shape[1] // 2]
        initials.append(initial_pos)


    return {'xy':xy_motions, 'z':z_motions, 'r':r_motions, 'pos':initials, 'f_num':frame_num}


def scaleObject(object, scale):
    if scale == 0:
        return object

    if scale == -1:
        return cv2.resize(object,None,fx=1-SCALE,fy=1-SCALE,interpolation=cv2.INTER_AREA)
    if scale == 1:
        return cv2.resize(object,None,fx=1+SCALE,fy=1+SCALE,interpolation=cv2.INTER_AREA)

    return object


def rotateObject(object, rotate):
    if rotate == 0:
        return object
    (h, w) = object.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    if rotate == 1:
        angle = ROTATE_ANGLE
    if rotate == -1:
        angle = -ROTATE_ANGLE
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(object, M, (nW, nH))


def putInObject(object, background, position):
    blur_mask = np.zeros((background.shape[0], background.shape[1]))
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

        alpha_ob = object[ob_y1:ob_y2, ob_x1:ob_x2, 3] / 255.0
        alpha_bg = 1.0 - alpha_ob
        for c in range(0, 3):
            background[bg_y1:bg_y2,bg_x1:bg_x2,c] = alpha_ob * object[ob_y1:ob_y2, ob_x1:ob_x2, c] + \
                                                    alpha_bg * background[bg_y1:bg_y2, bg_x1:bg_x2, c]

        blur_mask[bg_y1:bg_y2,bg_x1:bg_x2] = alpha_ob

    return background, blur_mask


def putInObjects(front_objects, background, motions):
    frames = []
    total_blur_mask = np.zeros((background.shape[0], background.shape[1])).astype(np.uint8)
    for f in range(motions['f_num']):
        bg = background.copy()
        for i,o in enumerate(front_objects):
            bg, blur_mask = putInObject(o, bg, motions['pos'][i])
            total_blur_mask = np.bitwise_or(total_blur_mask, blur_mask.astype(np.uint8))

            motions['pos'][i][1] += motions['xy'][i][f][1]
            motions['pos'][i][0] += motions['xy'][i][f][0]
            if motions['r'][i] != 0:
                front_objects[i] = rotateObject(o, motions['r'][i])
            if motions['z'][i] != 0:
                front_objects[i] = scaleObject(o, motions['z'][i])
        frames.append(bg)
    return frames, total_blur_mask



def getFrontImgBasedOnBackground(front_img_paths, background_path, front_object_limit=None):
    front_objects = []
    bg_img = cv2.imread(background_path)
    if bg_img.shape[2] == 1:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    for front_img_path in front_img_paths:
        front_img = cv2.imread(front_img_path)
        if front_img.shape[2] == 1:
            front_img = cv2.cvtColor(front_img, cv2.COLOR_GRAY2BGR)
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2BGRA)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
        # check grayscale img
        if front_img.shape[2] == 1:
            front_img = cv2.cvtColor(front_img, cv2.COLOR_GRAY2RGB)

        if bg_img.shape[2] == 1:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2RGB)
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2BGRA)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        front_masks = getFrontObjects(front_img_path, front_object_limit)

        for m in front_masks:
            front_i = front_img.copy()
            for c in range(4):
                front_i[:,:,c] = front_i[:,:,c] * m
            front_i = dePadImage(front_i)
            front_objects.append(front_i)

    motions = generateMotionPath(front_objects, bg_img.shape[0], bg_img.shape[1])
    frames, total_blur_mask = putInObjects(front_objects, bg_img, motions)
    return frames, total_blur_mask

def main():
    bg_test_ids = extractLabelList(COCO_ROOT)[:10000]
    f_test_ids = extractLabelList(COCO_ROOT)
    for bg_id in bg_test_ids:
        front_id = random.sample(f_test_ids, 2)
        front_id = [label2img(x) for x in front_id]
        bg_id = label2img(bg_id)
        frames, total_blur_mask = getFrontImgBasedOnBackground(front_id, bg_id)
        foldername = bg_id.split('/')[-1].split('.')[0]
        blur_frame = np.zeros(frames[0].shape).astype(np.float32)
        for (i,f) in enumerate(frames):
            blur_frame += f.astype(np.float32)
            cv2.imwrite('./{}.png'.format(i), f)
            # cv2.imwrite('./coco_blur_image/{}.png'.format(foldername), blur_frame)
        blur_frame = blur_frame / len(frames)
        blur_frame = blur_frame.astype(np.uint8)
        gt_index = len(frames) // 2 #21 //2 = 10 5 // 2 = 2 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        #cv2.imwrite('./coco_sharp_image/{}.png'.format(foldername), frames[gt_index])
        #cv2.imwrite('./coco_blur_image/{}.png'.format(foldername), blur_frame)
        #np.save('./coco_blur_mask/{}.npy'.format(foldername), total_blur_mask)
        cv2.imwrite('./{}.png'.format(foldername), blur_frame)
        plt.matshow(total_blur_mask)
        plt.savefig('./label.jpg')

        '''
        blur_c = blur_frame.copy()
        for c in range(4):
            blur_c[:, :, c] = blur_c[:, :, c] * total_blur_mask
        cv2.imwrite('./coco_check_image/{}.png'.format(foldername), blur_c)
        '''
        print('[Generating] {}, frame num: {}, gt index: {}'.format(foldername, len(frames), gt_index))

if __name__ == '__main__':
    main()

