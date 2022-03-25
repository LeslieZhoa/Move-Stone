import cv2
import numpy as np

# some PS blending

# 正片叠底
def Multiply (img_1, img_2):
    '''
    img_1,img_2 -> [0,1]
    '''
    img = img_1 * img_2
    return img

# 滤色
def Screen(img_1, img_2):
    '''
    img_1,img_2 -> [0,1]
    '''
    img = 1- (1-img_1)*(1-img_2)

    return img

# 柔光
def Soft_light(img_1, img_2):
    '''
    img_1,img_2 -> [0,1]
    '''
    mask = img_1 < 0.5
    T1 = (2 * img_1 -1)*(img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 -1)*(np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1-mask)

    return img






# read path contain chinese
path = ""
img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)[...,:3]
# ----------------------------compute the centroid from a mask image-------------------------
def compute_centroid(img_path):
    nThresh = 100
    nMaxThresh = 255
    if isinstance(img_path,str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.blur(gray, (3,3))

    cannyImg = cv2.Canny(gray, nThresh, nThresh*2, 3)
    contours, hierarchy = cv2.findContours(cannyImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mu = []
    mc = []
    retval = np.array([])
    for i in range(0, np.array(contours).shape[0]):
        retval = cv2.moments(contours[i], False)
        mu.append(retval)
    mu = np.array(mu)

    for i in range(0, np.array(contours).shape[0]):
        if mu[i]['m00'] == 0.0:
            a=0
            b=0
        else:
            a = mu[i]['m10'] / mu[i]['m00']  #centroid x coor
            b = mu[i]['m01'] / mu[i]['m00']  #centroid y coor

        

        mc.append([a,b])
    return np.array(mc) # the centroid

#---------------------------seamless----------------------------
import scipy.sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import cv2
import matplotlib.pyplot as plt
def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)    
    return mat_A
def seamless_clone(source, target, mask):
    '''
    Args:
        source: [h,w,3] value-->[0,1]
        target: [h,w,3] value-->[0,1]
        mask:   [h,w]   value-->[0,1]
        the mask area will erode before to get a better result 
        put source to target
    Return:
        [h,w,3] value-->[0,1]
    '''
    h, w,c = target.shape
    result = []
    
    mat_A = laplacian_matrix(h, w)
    laplacian = mat_A.tocsc()

    mask[0,:] = 1
    mask[-1,:] = 1
    mask[:,0] = 1
    mask[:,-1] = 1
    q = np.argwhere(mask==0)
    
    k = q[:,1]+q[:,0]*w
    mat_A[k, k] = 1
    mat_A[k, k + 1] = 0
    mat_A[k, k - 1] = 0
    mat_A[k, k + w] = 0
    mat_A[k, k - w] = 0

    mat_A = mat_A.tocsc()    
    mask_flat = mask.flatten()
    for channel in range(c):
        
        source_flat = source[:, :, channel].flatten()
        target_flat = target[:, :, channel].flatten()        

        mat_b = laplacian.dot(source_flat)*0.75
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b).reshape((h, w))
        result.append (x)

        
    return np.clip( np.dstack(result), 0, 1 )
#for example
source = cv2.imread('D:/code/STUDY/poisson-image-editing/temp/frame_img/66.png')[...,[2,1,0]]/255.0
target = cv2.imread('D:/code/STUDY/poisson-image-editing/temp/gen_img/66.png')[...,[2,1,0]]/255.0
mask = cv2.imread('D:/code/STUDY/poisson-image-editing/temp/mask_img/66.png',0)/255.0
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    
mask = cv2.erode(mask,kernel2)
r = seamless_clone(target,source,mask)
plt.imshow(r)
plt.show()
