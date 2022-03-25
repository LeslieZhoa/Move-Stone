import cv2
import os
import numpy as np
import math
import imutils
WHITE=[255,255,255]
'''
图像数据增强相关代码
'''
def rotate90(img_path):
    img=cv2.imread(img_path)
    add_or_sub=1
    if np.random.uniform()>0.5:
        add_or_sub=-1
    angle=int(add_or_sub*90)
    img=imutils.rotate_bound(img, angle)
    return img
def rotate(img_path, scale=1.):
    src=cv2.imread(img_path)
    
    w = src.shape[1]
    h = src.shape[0]
    add_or_sub=1
    if np.random.uniform()>0.5:
        add_or_sub=-1
    angle=int(np.random.randint(10,20)*add_or_sub)
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    rotate_img=cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,borderValue=(255,255,255))
    h1,w1,_=rotate_img.shape
    return rotate_img[(h1-h)//2:(h1+h)//2,(w1-w)//2:(w1+w)//2]


def flip(img_path):
    
    img=cv2.imread(img_path)

    img=cv2.flip(img,1)
    #             cv2.imshow('',img)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    return img

def random_crop_and_pad(img_path):
    
    img=cv2.imread(img_path)
    h,w,_=img.shape
    x1=int(np.random.randint(low=5,high=10)/100*h)

    y1=int(np.random.randint(low=10,high=20)/100*w)

    x2=int(np.random.randint(low=10,high=20)/100*h)
    y2=int(np.random.randint(low=5,high=10)/100*w)
    img=img[x1:h-x2,y1:w-y2]
    img=cv2.copyMakeBorder(img,x1,x2,y1,y2,cv2.BORDER_CONSTANT,value=WHITE)
#             cv2.imshow('',img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
    return img

