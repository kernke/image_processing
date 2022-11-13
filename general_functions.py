# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import numpy as np
import cv2
import copy
from numba import njit
import scipy.special

#%%
@njit
def make_circular_mask(x0,y0,r,image):   
    r2=r*r
    mask=np.zeros(np.shape(image))
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if ((i-x0)**2+(j-y0)**2) < r2:
                mask[i,j]=1
                 
    return mask


#%%
def get_max(z):
    maxpos=np.argmax(z)
    x0=maxpos//z.shape[1]
    y0=maxpos% z.shape[1]
    return np.array([x0,y0]).astype(int)

#%% pascal triangle
def pascal_numbers(n):
    '''Returns the n-th row of Pascal's triangle' '''
    return scipy.special.comb(n,np.arange(n+1))

#%% kernels

def smoothbox_kernel(kernel_size):
    '''Gaussian Smoothing kernel approximated by integer-values obtained via binomial distribution '''
    r=kernel_size[0]
    c=kernel_size[1]
    sb=np.zeros([r,c])
    
    row=pascal_numbers(r-1)
    col=pascal_numbers(c-1)
    
    row /= np.sum(row)
    col /= np.sum(col)
    
    for i in range(r):
        for j in range(c):
            sb[i,j]=row[i]*col[j]
            
    return sb

#%% rebin
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

#%% rotate    

#this function is a modified version of the original from 
#https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L41
def rotate_bound(image, angle,flag='cubic',flag2='forward',bm=1):
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int(np.round((h * sin) + (w * cos)))
    nH = int(np.round((h * cos) + (w * sin)))
    
    if bm== 0:    
        bm=cv2.BORDER_CONSTANT
    elif bm==1:
        bm=cv2.BORDER_REPLICATE
        
    if flag2 == 'back':
        if nW%2==1:
            nW +=1
        if nH%2==1:
            nH +=1
        

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    if flag =='cubic':   
        return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC,borderMode=bm)
    else:
        return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_LINEAR,borderMode=bm)
    
#%% make Mask

@njit
def make_mask(rot,d=3):
    mask=rot>0
    newmask=rot>0
    for i in range(d,mask.shape[0]-d):
        for j in range(d,mask.shape[1]-d):
            if mask[i,j]:

                if not mask[i-d,j] or not mask[i+d,j] or not mask[i,j-d] or not mask[i,j+d]:
                    newmask[i,j]=0
    
    newmask[:d,:]=0
    newmask[:,:d]=0
    newmask[-d:,:]=0
    newmask[:,-d:]=0
    return newmask


#%% asymmetric non maximum supppression

def anms(img,mask,thresh_ratio=1.5,ksize=5,asympix=0):
    newimg=copy.deepcopy(img)
    return aysmmetric_non_maximum_suppression(newimg,img,mask,thresh_ratio,ksize,asympix)

@njit
def aysmmetric_non_maximum_suppression(newimg,img,mask,thresh_ratio,ksize,asympix):
    ioffs=ksize//2#+asympix//2
    joffs=ksize//2+asympix//2
    #newimg=img#np.zeros(img.shape)
    for i in range(ioffs,img.shape[0]-ioffs):
        for j in range(joffs,img.shape[1]-joffs):
            if not mask[i,j]:
                pass
            elif not mask[i-ioffs,j-joffs] or not mask[i+ioffs,j+joffs] or not mask[i-ioffs,j+joffs] or not mask[i+ioffs,j-joffs]:
                newimg[i,j]=img[i,j]
            else:
                g=img[i-ioffs:i+ioffs+1,j-joffs:j+joffs+1]
                v=max(np.sum(g,axis=0)) #* ksize/(ksize+asympix)
                h=max(np.sum(g,axis=1)) * ksize/(ksize+asympix)
                if h>v*thresh_ratio:
                    newimg[i,j]=img[i,j]
                else:
                    newimg[i,j]=np.min(g)
    return newimg




#%% noise level determination from aysmmetric_non_maximum_suppression

@njit
def anms_noise(img,mask,thresh_ratio,ksize,asympix):
    ioffs=ksize//2#+asympix//2
    joffs=ksize//2+asympix//2
    npix=ksize*(ksize+asympix)
    noisemean=[]
    noisemax=[]
    noisestd=[]
    #newimg=img#np.zeros(img.shape)
    for i in range(ioffs,img.shape[0]-ioffs):
        for j in range(joffs,img.shape[1]-joffs):
            if not mask[i,j]:
                pass
            elif not mask[i-ioffs,j-joffs] or not mask[i+ioffs,j+joffs] or not mask[i-ioffs,j+joffs] or not mask[i+ioffs,j-joffs]:
                pass
                #    newimg[i,j]=img[i,j]
            else:
                g=img[i-ioffs:i+ioffs+1,j-joffs:j+joffs+1]
                v=max(np.sum(g,axis=0)) #* ksize/(ksize+asympix)
                h=max(np.sum(g,axis=1)) 
                if h* ksize/(ksize+asympix)  >  v*thresh_ratio:
                    pass
                    #newimg[i,j]=img[i,j]
                else:
                    ave=(v+h)/npix
                    g=(g-ave)**2
                    noisemean.append(ave)
                    noisemax.append(np.max(g))
                    std=np.sum(g)
                    noisestd.append(std)
                    #newimg[i,j]=np.min(g)
    return noisemax,noisemean,noisestd



def determine_noise_threshold(img,mask,thresh_ratio,ksize,asympix):
    npix=ksize*(ksize+asympix)
    
    noisemax,noisemean,noisestd=anms_noise(img,mask,thresh_ratio,ksize,asympix)
    nma=np.array(noisemax)
    nme=np.array(noisemean)
    nms=np.array(noisestd)
    return np.sqrt(nma),nme,np.sqrt(nms/npix)