import cv2
import numpy as np
import os
# dmc is pycocotools
def subimage(image, center, theta, width, height):

    ''' 
    https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    '''

    # Uncomment for theta in radians
    #theta *= 180/np.pi

    im_shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
    image = cv2.warpAffine( src=image, M=matrix, dsize=im_shape )

    x = int( center[0] - width/2  )
    y = int( center[1] - height/2 )

    image = image[ y:y+height, x:x+width ]

    return image

def ann_based_threshold(img_idx):
    img = dmc.loadImgs(img_idx)[0]
    img_path = os.path.join(data_dir, "images", img["file_name"])
    im = cv2.imread(img_path)
    annIds = dmc.getAnnIds(imgIds=img_idx, iscrowd=None)
    anns = dmc.loadAnns(annIds)
    thres = []
    for ann in anns:
        bbox= ann['bbox']
        x,y,w,h = np.where(np.array(bbox)<0,0,bbox)
        rotate = ann['attributes']['rotation']
        center_x = x + w//2
        center_y = y + h//2
        coordinates = ((center_x,center_y),(w,h),rotate)
        box = cv2.boxPoints(coordinates)
        box = np.int0(box)
        cx = np.float32(np.int_(center_x))
        cy = np.float32(np.int_(center_y))
        w = np.int_(w) ; h = np.int_(h)
        if ann['category_id'] == 2:
            crop = subimage(im,(cx,cy),rotate,w,h)
            thres.append(np.nanmean(crop))
        elif ann['category_id'] == 3:
            crop = subimage(im,(cx,cy),rotate,w,h)
            thres.append(np.nanmean(crop))
        elif ann['category_id'] == 4:
            crop = subimage(im,(cx,cy),rotate,w,h)
            thres.append(np.nanmean(crop))
    return np.nanmean(thres)