import cv2
def synthetic_otsu_background(img,background):
    img = cv2.imread(img) # original image
    background = cv2.imread(background) # cut image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    t, t_otsu = cv2.threshold(gray, -1, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = t_otsu.copy()
    
    return cv2.copyTo(img,mask,background)