import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.patches as patches
import random
from glob import glob
# dmc is pycocotools

def myLoadImage(i,img_size, augment=True) :
    img = dmc.loadImgs(i)[0]
    f = os.path.join(data_dir, "images", img["file_name"])
    im = cv2.imread(f)
    img_size = im.shape[1]
    
    h0, w0 = im.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if augment else cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return im, (h0, w0), im.shape[:2] 
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    # if clip:
    #     clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2)/w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) /h # y center
    y[:, 2] = (x[:, 2] - x[:, 0])/w  # width
    y[:, 3] = (x[:, 3] - x[:, 1])/h  # height
    return y
def noPad(label,s, xc, yc, xmin, ymin) :
    label4_ = label.copy()
    if ((xc>s) & (yc<s)) :
        label4_[:,1] = label4_[:,1] - xmin
        label4_[:,3] = label4_[:,3] - xmin
    elif ((xc<s) & (yc>s)) :
        label4_[:,2] = label4_[:,2] - ymin
        label4_[:,4] = label4_[:,4] - ymin
    elif ((xc>s) & (yc>s)) :
        label4_[:,1] = label4_[:,1] - xmin
        label4_[:,3] = label4_[:,3] - xmin
        label4_[:,2] = label4_[:,2] - ymin
        label4_[:,4] = label4_[:,4] - ymin
    return label4_

def yolo_label(coco_xywh,w,h):
    yolo_xywh = coco_xywh.copy()
    yolo_xywh[:,0] = (coco_xywh[:,0] + (coco_xywh[:,2]/2))/w # noraml center x
    yolo_xywh[:,1] = (coco_xywh[:,1] + (coco_xywh[:,3]/2))/h # normal center y
    yolo_xywh[:,2] = coco_xywh[:,2]/w
    yolo_xywh[:,3] = coco_xywh[:,3]/h
    return yolo_xywh

def yolo2coco(yolo_xywhn,w,h):
    coco_xywh = yolo_xywhn.copy()
    coco_xywh[:,0] = (yolo_xywhn[:,0]- (yolo_xywhn[:,2]/2)) * w # minx
    coco_xywh[:,1] = (yolo_xywhn[:,1]- (yolo_xywhn[:,3]/2)) * h # minx
    coco_xywh[:,2] = yolo_xywhn[:,2] * w
    coco_xywh[:,3] = yolo_xywhn[:,3] * h
    return coco_xywh

def cocoMosaic(img_size, path, form = "coco") :
    labels4 = []
    s = img_size
    mosaic_border = [-img_size // 2, -img_size // 2]

    # center point
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y # 81 172

    # image files and labels
    # train set & not Scratch
    
    
    im_files = sorted(glob.glob(path+'/*.png'))
    
    
#     label_files = img2label_paths(im_files)  # 얘 수정해야됨

    n = len(im_files)
    indices_ = range(n)

    # random 4 images
    indices = random.choices(indices_, k=4)  # 3 additional image indices # [0, 3292, 20762, 18713]
    random.shuffle(indices)  # [18713, 0, 20762, 3292]
    print(indices)

    for i, index in enumerate(indices):
        # Load image
        img, (h0, w0), (h, w) = myLoadImage(index,s)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

            xmin, ymin = x1a, y1a

        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            xmax, ymax = x2a, y2a

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        annIds = dmc.getAnnIds(imgIds=index, iscrowd=None) # catIds can be given.
        anns = dmc.loadAnns(annIds)
        
        
        labels = np.array([[ann['category_id']]+ann['bbox'] for ann in anns])
        labels[:,1:] = yolo_label(labels[:,1:],w,h)
#         labels = np.array([ann['bbox'] for ann in anns])
        if len(labels):
            # coco is xywh (not normalize)
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels[labels < 0] = 0
            labels[labels > s*2] = s*2
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        labels4[:, 1:] = np.clip(labels4[:, 1:], 0, 2 * s)  # use with random_affine
#     labels4 = np.concatenate(labels4, 0)

    # no padding
    img4 = img4[ymin:ymax, xmin:xmax,:]
    labels4 = noPad(labels4, s,xc, yc, xmin, ymin)
    # resize
    img4 = cv2.resize(img4, (w,h))
    labels4[:,1:] = xyxy2xywhn(labels4[:,1:], w = xmax - xmin, h = ymax-ymin)
    labels4 = labels4[labels4[:,3]>0]
    labels4 = labels4[labels4[:,4]>0]
    
    if form == "xyxy" :
        labels4[:,1:] = xywhn2xyxy(labels4[:,1:], w = img4.shape[1], h = img4.shape[0])
    elif form == 'coco':
        labels4[:,1:] = yolo2coco(labels4[:,1:], w = img4.shape[1], h = img4.shape[0])
    return img4, labels4