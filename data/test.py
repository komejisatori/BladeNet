import numpy as np
import glob
import matplotlib.pyplot as plot
thres = 0.25

for f in glob.glob("/dataset/RealData/localval/optical_mask/*.npy"):
    id = f.split('/')[-1].split('.')[0]
    mask = np.load(f, allow_pickle=True)
    indexs1 = np.where(mask <= thres)
    flow_mask1 = np.zeros(mask.shape[:2])
    flow_mask1[indexs1] = 1
    indexs2 = np.where(mask > thres)
    flow_mask2 = np.zeros(mask.shape[:2])
    flow_mask2[indexs2] = 1
    plot.matshow(flow_mask1)
    plot.savefig("/dataset/RealData/localval/thres_below25/"+id+".jpg")
    plot.close()
    plot.matshow(flow_mask2)
    plot.savefig("/dataset/RealData/localval/thres_above25/" + id + ".jpg")
    plot.close()
