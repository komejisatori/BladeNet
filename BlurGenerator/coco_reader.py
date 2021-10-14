import os
import numpy as np
from pycocotools import mask
from pycocotools.cocostuffhelper import getCMap
from pycocotools.coco import COCO
from PIL import Image, ImagePalette
import skimage.io
import matplotlib.pyplot as plt

def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    labelMap = np.zeros(imageSize)

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        #labelMask = labelMasks[:, :, a] == 1
        newLabel = imgAnnots[a]['category_id']

        if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

        labelMap[labelMask] = newLabel

    return labelMap


def cocoSegmentationToPng(coco,imgId,pngPath, includeCrowd=False):
    labelMap = cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)
    labelMap = labelMap.astype(np.int8)
    # Get color map and convert to PIL's format
    cmap = getCMap()
    cmap = (cmap * 255).astype(int)
    padding = np.zeros((256-cmap.shape[0], 3), np.int8)
    cmap = np.vstack((cmap, padding))
    cmap = cmap.reshape((-1))
    cmap = np.uint8(cmap).tolist()
    assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

    # Write to png file
    png = Image.fromarray(labelMap).convert('P')
    png.putpalette(cmap)
    png.save(pngPath, format='PNG')


def cocoSegmentationToPngDemo(data_dir='/home/wangzerun/data/new_coco',data_type='train2017',
                              pngFolderName='./',isAnnotation=True):
    annPath = os.path.join(data_dir,'annotations/stuff_anno/stuff_{}.json'.format(data_type))

    coco = COCO(annPath)
    imgIds = coco.getImgIds()
    imgCount = len(imgIds)
    for i in range(imgCount):
        imgId = imgIds[i]
        imgName = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.jpg', '')
        print('Exporting image %d of %d: %s' % (i + 1, imgCount, imgName))
        segmentationPath = '%s/%s.png' % (pngFolderName, imgName)
        cocoSegmentationToPng(coco, imgId, segmentationPath)


if __name__ == '__main__':
    cocoSegmentationToPngDemo()

