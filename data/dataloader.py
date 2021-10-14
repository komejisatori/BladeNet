import random
import numpy as np
import torch
import cv2
import torch.utils.data as data
import data.util as util


class MaskDataset(data.Dataset):

    def __init__(self, sharp_root, blur_root, mask_root, resize_size, patch_size, phase='train'):
        
        super(MaskDataset).__init__()
        self.n_channels = 3
        self.resize_size = resize_size
        self.patch_size = patch_size
        self.phase = phase
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(sharp_root)
        self.paths_L = util.get_image_paths(blur_root)
        self.paths_M = util.get_image_paths(mask_root)

        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        #H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        M_path = self.paths_M[index]

        #img_H = util.imread_uint(H_path, self.n_channels)
        
        mask = np.load(M_path, allow_pickle = True)

        img_L = util.imread_uint(L_path)
        

        if self.phase == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            assert self.resize_size is 0
            if self.patch_size != 0:
                H, W, C = img_L.shape

                # --------------------------------
                # randomly crop L patch
                # --------------------------------
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
                img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

                # --------------------------------
                # crop corresponding H patch
                # --------------------------------
                #rnd_h_H, rnd_w_H = int(rnd_h), int(rnd_w)
                #img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
                mask = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            img_L= util.augment_img(img_L, mode=mode)
            mask = util.augment_img(mask,mode=mode)
            # --------------------------------
            # get patch pairs
            # --------------------------------
            #img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_L = util.single2tensor3(img_L)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
            
        else:
            img_L = util.uint2single(img_L)
            img_L = util.single2tensor3(img_L)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        


        return {'L': img_L, 'mask':mask, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)





class AugDataset(data.Dataset):
    '''
        # -----------------------------------------
        # Get L/H/M for noisy image SR.
        # Only "paths_H" is needed, sythesize bicubicly downsampled L on-the-fly.
        # -----------------------------------------
        # e.g., SRResNet super-resolver prior for DPSR
        # -----------------------------------------
        '''

    def __init__(self, sharp_root, blur_root, mask_root, resize_size, patch_size, phase='train'):
        super(AugDataset).__init__()

        self.n_channels = 3
        self.resize_size = resize_size
        self.patch_size = patch_size
        self.phase = phase
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(sharp_root)
        self.paths_L = util.get_image_paths(blur_root)
        self.paths_M = util.get_image_paths(mask_root)

        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        M_path = self.paths_M[index]

        img_H = util.imread_uint(H_path, self.n_channels)
        
        mask = np.load(M_path, allow_pickle = True)

        img_L = util.imread_uint(L_path)
        

        if self.phase == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            assert self.resize_size is 0
            if self.patch_size != 0:
                H, W, C = img_L.shape

                # --------------------------------
                # randomly crop L patch
                # --------------------------------
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
                img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

                # --------------------------------
                # crop corresponding H patch
                # --------------------------------
                rnd_h_H, rnd_w_H = int(rnd_h), int(rnd_w)
                img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
                mask = mask[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
            mask = util.augment_img(mask,mode=mode)
            # --------------------------------
            # get patch pairs
            # --------------------------------
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
            
        else:
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        


        return {'L': img_L, 'H': img_H, 'mask':mask, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class OpticalDataset(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H/M for noisy image SR.
    # Only "paths_H" is needed, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet super-resolver prior for DPSR
    # -----------------------------------------
    '''

    def __init__(self, sharp_root1, blur_root1, mask_root1, resize_size, patch_size, phase='train'):
        super(OpticalDataset).__init__()

        self.n_channels =  3
        self.resize_size = resize_size
        self.patch_size = patch_size
        self.phase = phase
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(sharp_root1)
        self.paths_L = util.get_image_paths(blur_root1)
        self.paths_OP = util.get_image_paths(mask_root1)

        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        M_path = self.paths_OP[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        mask = np.load(M_path, allow_pickle=True)

        # ------------------------------------
        # sythesize L image via matlab's bicubic
        # ------------------------------------
        H, W, _ = img_H.shape
        img_L = util.imread_uint(L_path, self.n_channels)

        assert self.phase == 'test'

        img_H = util.uint2single(img_H)
        img_L = util.uint2single(img_L)
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)


        return {'L': img_L, 'H': img_H, 'mask': mask, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class OriDataset(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H/M for noisy image SR.
    # Only "paths_H" is needed, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet super-resolver prior for DPSR
    # -----------------------------------------
    '''

    def __init__(self, sharp_root1, blur_root1, sharp_root2, blur_root2, resize_size, patch_size, phase='train'):
        super(OriDataset).__init__()

        self.n_channels =  3
        self.resize_size = resize_size
        self.patch_size = patch_size
        self.phase = phase
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(sharp_root1)
        self.paths_L = util.get_image_paths(blur_root1)
        if sharp_root2 is not None and sharp_root2 != 'None':
            self.paths_H = self.paths_H + util.get_image_paths(sharp_root2)
        if blur_root2 is not None and blur_root2 != 'None':
            self.paths_L = self.paths_L + util.get_image_paths(blur_root2)

        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        img_H = util.imread_uint(H_path, self.n_channels)


        # ------------------------------------
        # sythesize L image via matlab's bicubic
        # ------------------------------------
        H, W, _ = img_H.shape
        img_L = util.imread_uint(L_path, self.n_channels)


        if self.phase == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            if self.resize_size != 0:
                img_L = cv2.resize(img_L, (self.resize_size, self.resize_size))
                img_H = cv2.resize(img_H, (self.resize_size, self.resize_size))
            if self.patch_size != 0:
                H, W, C = img_L.shape

                # --------------------------------
                # randomly crop L patch
                # --------------------------------
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
                img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

                # --------------------------------
                # crop corresponding H patch
                # --------------------------------
                rnd_h_H, rnd_w_H = int(rnd_h), int(rnd_w)
                img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            #mode = np.random.randint(0, 8)
            #img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

            # --------------------------------
            # get patch pairs
            # --------------------------------
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        else:
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)


        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    import glob
    import shutil
    #TRAIN_SHARP = '/dataset/REDS_yf/train/sharp' #0.4831
    #TRAIN_BLUR = '/dataset/REDS_yf/train/blur'
    TRAIN_SHARP = '/dataset/REDS_AUG/train/sharp/'
    TRAIN_BLUR = '/dataset/REDS_AUG/train/sharp/'
    TEST_SHARP = "/dataset/RealData/localval/sharp_optical/"
    TEST_BLUR = "/dataset/RealData/localval/blur_optical/"
    TEST_MASK = "/dataset/RealData/localval/optical_mask/"
    trainSet = OpticalDataset(sharp_root1=TEST_SHARP, blur_root1=TEST_BLUR,
                            mask_root1=TEST_MASK,
                          patch_size=100, resize_size=0, phase='test')
    train_loader = DataLoader(trainSet,
                              batch_size=64,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,
                              pin_memory=True)
    avg_time = 0
    start = time.time()
    idx = 0
    for i, train_data in enumerate(train_loader):
        end = time.time()
        avg_time += end - start
        print(end - start)
        start = end
        idx += 1

    print(avg_time / idx)
