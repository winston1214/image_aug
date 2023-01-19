# -*- coding: utf-8 -*-
from PIL import Image, ImageOps
import torchvision.transforms as T
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import glob
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import os

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    
    random_state = np.random.RandomState(random_state)
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def background_aug(img_dir,target_dir,elastic_num=10): # head skin image augmentation
    img_ls = sorted(glob.glob(img_dir+'*'))
    for i in img_ls: # jpg to png
        if '.jpg' in i:
            png_name = i.replace('.jpg','.png')
            im = Image.open(i).convert('RGB')
            im.save(png_name)
            os.remove(i)
    img_ls = sorted(glob.glob(img_dir+'*')) # img_ls redefine
    for i in tqdm(img_ls): # 1차 원본 elastic transform
        new_img_ls = []
        name = i.split('/')[-1]
        img = Image.open(i).convert('RGB')
        ### elastic transform 을 위한 bgr로 변환
        cv2_img = np.array(img)[:,:,::-1]
        for j in range(elastic_num):
            ela_img = elastic_transform(cv2_img, cv2_img.shape[1] * 2, cv2_img.shape[1] * 0.08, cv2_img.shape[1] * 0.08,random_state = j)
            rgb = cv2.cvtColor(ela_img,cv2.COLOR_BGR2RGB)
            ela_pil_img = Image.fromarray(rgb)
            new_img_ls.append(ela_pil_img)

            ## Elastic transform -> flip
            v_flip = ImageOps.flip(ela_pil_img).convert('RGB') # vertical flip
            h_flip = ImageOps.mirror(ela_pil_img).convert('RGB') # horizontal flip
            new_img_ls.append(v_flip)
            new_img_ls.append(h_flip)
            
            ## Elastic Transform -> ColorJitter
            for b in np.arange(0.1,0.5,0.05):
                color_aug = T.ColorJitter(brightness=(0.6,1.2))
                color_aug1 = T.ColorJitter(brightness=(0.6,1.2),contrast=b)
                color_aug2 = T.ColorJitter(brightness=(0.6,1.2),saturation=b)
                color_aug3 = T.ColorJitter(brightness=(0.6,1.2),contrast=b,saturation=b)
                color_aug4 = T.ColorJitter(contrast=b)
                color_aug5 = T.ColorJitter(saturation=b)
                
                new_img_ls.append(color_aug(ela_pil_img).convert('RGB'));new_img_ls.append(color_aug1(ela_pil_img).convert('RGB'));
                new_img_ls.append(color_aug2(ela_pil_img).convert('RGB'));new_img_ls.append(color_aug3(ela_pil_img).convert('RGB'));
                new_img_ls.append(color_aug4(ela_pil_img).convert('RGB'));new_img_ls.append(color_aug5(ela_pil_img).convert('RGB'));
        ## original -> Transform
        org_v_flip = ImageOps.flip(img).convert('RGB') # vertical flip
        org_h_flip = ImageOps.mirror(img).convert('RGB') # horizontal flip
        
        for b in np.arange(0.1,0.5,0.05):
            org_color_aug = T.ColorJitter(brightness=(0.6,1.2))
            org_color_aug1 = T.ColorJitter(brightness=(0.6,1.2),contrast=b)
            org_color_aug2 = T.ColorJitter(brightness=(0.6,1.2),saturation=b)
            org_color_aug3 = T.ColorJitter(brightness=(0.6,1.2),contrast=b,saturation=b)
            org_color_aug4 = T.ColorJitter(contrast=b)
            org_color_aug5 = T.ColorJitter(saturation=b)
        new_img_ls.append(org_v_flip);new_img_ls.append(org_h_flip);
        new_img_ls.append(org_color_aug(img).convert('RGB'));new_img_ls.append(org_color_aug1(img).convert('RGB'));
        new_img_ls.append(org_color_aug2(img).convert('RGB'));new_img_ls.append(org_color_aug3(img).convert('RGB'));
        new_img_ls.append(org_color_aug4(img).convert('RGB'));new_img_ls.append(org_color_aug5(img).convert('RGB'));
        
        ## img save
        for idx,im in enumerate(new_img_ls):
            im.save(target_dir+name.replace('.png',f'_aug{idx}.png'))
                
    print('Augmentation Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',type=str,help='image folder')
    parser.add_argument('--save_dir',type=str,help='image folder')
    parser.add_argument('--num',type=int,help='elastic num',default=10)
    args = parser.parse_args()

    img_path = args.img_dir
    target_path = args.save_dir
    if img_path[-1] != '/':
        img_path += '/'
    if target_path[-1] != '/':
        target_path += '/'
        
    background_aug(img_path, target_path,args.num)
