import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def gray_nonlinear(img):
    ## img = cv2 imread
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # opencv grayscaling parameter
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]) # nonlinear method
    nonlinear = 255 * (gray / 255) ** (1/2.2)
    return nonlinear

def gray_avg(img):
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    avg = np.mean(rgb,axis=2)
    return avg

def gray_lightness(img):
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    rgb_max = np.max(rgb, axis=2)
    rgb_min = np.min(rgb, axis=2)
    lightness = (rgb_max + rgb_min)/2
    return lightness

def visualization_gray(img_path,show='all'):
    img = cv2.imread(img_path)
    nonlinear = gray_nonlinear(img)
    avg = gray_avg(img)
    light = gray_lightness(img)
    opencv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dic = {'Opencv Gray':opencv,'Lightness':light,'AVG':avg,'nonlinear':nonlinear}
    plt.figure(figsize=(20,12))
    if show == 'all':
        for idx,(k,v) in enumerate(dic.items()):
            plt.subplot(2,2,idx+1)
            plt.imshow(v,cmap='gray')
            plt.title(k)
    elif show == 'Light':
        plt.imshow(light,cmap='gray')
        plt.title('Lightness')
    elif show == 'opencv':
        plt.imshow(opencv,cmap='gray')
        plt.title('Opencv Gray')
    elif show == 'avg':
        plt.imshow(avg,cmap='gray')
        plt.title('AVG')
    elif show == 'nonlinear':
        plt.imshow(nonlinear,cmap = 'gray')
        plt.title('Nonlinear')
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',type=str,help='Image file name')
    args = parser.parse_args()
    visualization_gray(args.img_path)
    plt.close()
    