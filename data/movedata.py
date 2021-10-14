TAR_PATH = '/dataset/REDS_AUG/train/sharp/'

ORI_PATH = "/home/wangzerun/BlurGenerator/Generator/voc_sharp_image/"

import glob
import shutil
files = glob.glob(ORI_PATH+'*.bmp')
for f in files:
    src_imgname = f.split('/')[-1]
    tar_imgname = 'aug'+src_imgname
    print(tar_imgname)
    shutil.copy(ORI_PATH+src_imgname, TAR_PATH+tar_imgname)

