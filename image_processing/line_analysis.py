# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from .general_functions import *

#%% obtain snr  for optimal rotation

def obtain_snr(image,mask,line,show,minlength):
    
    rmeans=[]
    rstds=[]
    
    for j in range(image.shape[0]):
        roi=image[j][mask[j]]

        if len(roi)<minlength:
            pass
        else:
            rmeans.append(np.mean(roi))
            rstds.append(np.mean(np.sqrt((roi-rmeans[-1])**2))/np.sqrt(len(roi)))
                
    rmeans=np.array(rmeans)
    x=np.arange(len(rmeans))
    p=np.polyfit(x,rmeans,5)
    rmeans-=np.polyval(p,x)            
    rstds=np.array(rstds)
    
    
    val=rmeans/rstds
    if line== 'dark':
        val = np.mean(val)-val
    elif line=='bright':
        val -= np.mean(val)
        
    vstd=np.std(val)

    if show:
        val2=val-np.min(val)      
        plt.plot(val2 *1/np.max(val2 ),c='r')
        plt.show()    
    
    return np.max(val)/vstd
    


#%% optimal rotation

def optimal_rotation(image_roi,angle,thresh=5,minlength=50,line='dark',show=False):
    #image_roi should contain lines with the respective angle, optimally not ending inside the ROI
    #angle in degree
    
    #optimal angle is determined by the two characteristics of a misfit line:
    #lower brightness
    #lower brightness-variance
    
    pmangle=np.arange(-3,4)
    
    dummy=np.ones(np.shape(image_roi))
    
    for k in range(5):
        snr=np.zeros(7)
        for i in range(7):
            rot=rotate_bound(image_roi,angle+pmangle[i])
            drot=rotate_bound(dummy,angle+pmangle[i],bm=0)
            mask=drot>0.1
            #rot -= pr/5

            snr[i]=obtain_snr(rot, mask, line,False,minlength)
            
        #plt.plot(snr)
        #plt.show()
        
        if np.max(snr)<thresh:
            print('signal to noise ratio below threshold')
            return False
        
        angle+=pmangle[np.argmax(snr)]
        pmangle=pmangle/3
        
    res=np.round(angle,2)
    
    if show:
        rot=rotate_bound(image_roi,res)
        drot=rotate_bound(dummy,res,bm=0)
        mask=drot>0.1
        obtain_snr(rot, mask, line,True,minlength)
        plt.imshow(rot*mask)
        plt.show()

    return res



#%% centered circular plot

# deorecated
def center_circ_show(img,radius=0,reduce=False):
    shape=img.shape
    center=np.array(shape)/2
    if radius==0:
        radius=np.min(center)
    polar=cv2.warpPolar(img,center=center[::-1],maxRadius=radius,dsize=(0,0),flags=cv2.WARP_POLAR_LINEAR )
    invpolar=cv2.warpPolar(polar,center=center[::-1],dsize=shape,maxRadius=radius,flags=cv2.WARP_POLAR_LINEAR +cv2.WARP_INVERSE_MAP )
    row_lim=[0,shape[0]]
    col_lim=[0,shape[1]]
    
    if reduce:
        rows=np.sum(invpolar,axis=1)
        cols=np.sum(invpolar,axis=0)
        for i in range(len(rows)):
            if rows[i]!=0:
                row_lim[0]=i
                break
        for i in range(len(rows)):
            if rows[len(rows)-1-i]!=0:
                row_lim[1]=len(rows)-i
                break
        for i in range(len(cols)):
            if cols[i]!=0:
                col_lim[0]=i
                break
        for i in range(len(cols)):
            if rows[len(cols)-1-i]!=0:
                col_lim[1]=len(cols)-i
                break
   
    return invpolar[row_lim[0]:row_lim[1],col_lim[0]:col_lim[1]]


#%% Enhance Lines
def enhance_lines(image,angle,number_of_bins=61,ksize=None,dist=1,iterations=2,line='dark'):
    dummy=np.ones(image.shape)
    rot=rotate_bound(image, angle)
    drot=rotate_bound(dummy, angle,bm=0)
    newmask=make_mask(drot,4)
    

    trot=np.clip(rot,np.min(image),np.max(image))
    if line == 'dark':
        trot -= np.min(trot)
        trot =np.max(trot)-trot
    elif line == 'bright':
        pass
    
    trot =trot/np.max(trot)*255
    
    resl=[]
    resl.append(trot*newmask)


    tres=trot
    res=trot

    for i in range(iterations):    
        if ksize is None:
            srot=cv2.Sobel(tres,cv2.CV_64F,0,1)
        else:
            srot=cv2.Sobel(tres,cv2.CV_64F,0,1,ksize=ksize)


        srot-=np.min(srot)
        srot*=newmask
        
        flatsrot=srot.reshape(srot.shape[0]*srot.shape[1])
        counts,bins1=np.histogram(flatsrot,100)
        maxi=np.argmax(counts[1:])+1
        maxpos=maxi*bins1[1]
    
        histrange=1.2
        bins=np.linspace(maxpos/histrange,histrange*maxpos,number_of_bins)
        counts,bins2=np.histogram(flatsrot,bins=bins)
        
        bincenters=bins2[1:]-(bins2[1]-bins2[0])/2
        
        middle=np.sum(counts*bincenters)/np.sum(counts)
         
        t1=srot>middle
        t2a=srot<=middle
        t2b=srot>0
        t2=t2a*t2b
    
        tres=np.zeros(srot.shape)
        tres[:-dist,:]+=t2[dist:,:]*np.abs(srot[dist:,:]-middle)
        tres[dist:,:]+=t1[:-dist,:]*np.abs(srot[:-dist,:]-middle)

        tres=tres/np.max(tres) *255
        res *= tres
        resl.append(tres*newmask)
    
    return res**(1/(iterations+1))*newmask,resl,newmask,drot #trot3* #srot #drot


