import random
import numpy as np
import torch
import cv2
import torch.utils.data as data
from torch.utils.data import DataLoader
import Generator.NewBlurGenerator.dataset_util as util


class GoProDatasetSplit(data.Dataset):

    def __init__(self, sharp_root, mask_root, resize_size=0, patch_size=[240, 320], phase='train'):
        super(GoProDatasetSplit).__init__()

        self.n_channels = 3
        self.resize_size = resize_size
        self.patch_size = patch_size
        self.phase = phase
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(sharp_root)
        self.paths_M = util.get_image_paths(mask_root)
        #
        self.imgs_H = []
        self.imgs_M = []

        for H_path in self.paths_H:
            self.imgs_H.append(util.imread_uint(H_path, self.n_channels))
        for M_path in self.paths_M:
            self.imgs_M.append(np.load(M_path))

        print("{} {} train images".format(phase,len(self.paths_H)))
        assert self.paths_H, 'Error: Sharp path is empty.'
        assert self.paths_M, 'Error: Blur path is empty.'

    def __getitem__(self, index):

        if self.phase == 'train':
            #img_H = util.imread_uint(self.paths_H[index], self.n_channels)
            #img_L = util.imread_uint(self.paths_L[index], self.n_channels)
            img_H = self.imgs_H[index]
            img_M = self.imgs_M[index]

            H, W, _ = img_H.shape

            if self.patch_size != 0:
                # --------------------------------
                # randomly crop L patch
                # --------------------------------
                rnd_h = random.randint(0, max(0, H - self.patch_size[0]))
                rnd_w = random.randint(0, max(0, W - self.patch_size[1]))
                img_M = img_M[rnd_h:rnd_h + self.patch_size[0], rnd_w:rnd_w + self.patch_size[1], :]
                img_H = img_H[rnd_h:rnd_h + self.patch_size[0], rnd_w:rnd_w + self.patch_size[1], :]

            level = np.sum(img_M) / (self.patch_size[0] * self.patch_size[1])
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            #mode = np.random.randint(0, 8)
            #img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        return {'L': level, 'H': img_H}

    def __len__(self):
        return len(self.paths_H)


if __name__ =='__main__':
    blurData = GoProDatasetSplit(sharp_root='/dataset/GoPro_wzr/test/sharp', blur_root='/dataset/GoPro_wzr/test/blur',
                                 phase='test')
    train_loader = DataLoader(blurData,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              drop_last=False,
                              pin_memory=False)
    avg_time = 0

    idx = 0
    for i, train_data in enumerate(train_loader):
        import pdb
        pdb.set_trace()
        img_L = train_data['L']
        img_H = train_data['H']
        img_L = util.tensor2uint(img_L)
        img_H = util.tensor2uint(img_H)
        util.imsave(img_L, './testL_{}.png'.format(idx))
        util.imsave(img_H, './testH_{}.png'.format(idx))
        idx += 1