
# coding: utf-8

# In[ ]:

# version 0 works for the MIT experimental set up
#from ipsh import ipsh
import numpy as np
import cv2
from time import clock
import csv
import os

MHI_DURATION = 1
MAX_TIME_DELTA = 0.5 # No usage for now
MIN_TIME_DELTA = 0.05 # No usage for now
THRESH_VALUE = 32

# This function aims to extract features for various activities 
def video_feature_extraction_save(videoName, featureWriter, maskRegion, case, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE,DISPLAY=False): 
    cv2.namedWindow('rat activity recognition')
    visuals = ['input', 'frame_diff', 'motion_hist', 'grad_orient']
    # use MHI features (motion history intensity)
    visual_name = visuals[2]
  
    cam = cv2.VideoCapture(videoName)
    video_len = cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    ret, frame = cam.read()
  
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:,:,1] = 255
    ii = 0         
    while (ii<video_len-1):
        ii += 1
        ret, frame = cam.read()
            
        frame_diff = cv2.absdiff(frame, prev_frame)      
        
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)*maskRegion 
     
        ret, motion_mask = cv2.threshold(gray_diff, THRESH_VALUE, 1, cv2.THRESH_BINARY)
        timestamp = clock()
        cv2.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        mg_mask, mg_orient = cv2.calcMotionGradient(motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5 )
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)

        if visual_name == 'input':
            vis = frame.copy()
        elif visual_name == 'frame_diff':
            vis = frame_diff.copy()
        elif visual_name == 'motion_hist':
            vis0 = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            junk,mei0 = cv2.threshold(vis0,1,255,cv2.THRESH_BINARY)

        elif visual_name == 'grad_orient':
            hsv[:,:,0] = mg_orient/2
            hsv[:,:,2] = mg_mask*255
            vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        ## Compute features
        M1 = cv2.moments(mei0)
        M2 = cv2.moments(vis0)    
        Hu1 = cv2.HuMoments(M1)
        Hu2 = cv2.HuMoments(M2)
        
        smallNum = [1e-200]*7 
        Hu1 = Hu1 + smallNum
        Hu2 = Hu2 + smallNum
        
        Hu1 = np.sign(Hu1)*np.log10(np.abs(Hu1))
        Hu2 = np.sign(Hu2)*np.log10(np.abs(Hu2))
        
        if M1['m00']!=0:
            cx1 = M1['m10']/M1['m00']
            cy1 = M1['m01']/M1['m00']
        else:
            cx1 = 0;
            cy1 = 0; 
            
        if M2['m00']!=0:
            cx2 = M2['m10']/M2['m00']
            cy2 = M2['m01']/M2['m00']
        else:
            cx2 = 0;
            cy2 = 0;     
                                       
        meiSize = np.count_nonzero(mei0);
        
        if meiSize == 0:
            corner1 = 0
            corner2 = 0
            corner3 = 0
            corner4 = 0
            height = 0
            width = 0
            extend = 0
        else:
            maskInd = np.nonzero(maskRegion)
            maskCx = np.mean(maskInd[0])
            maskCy = np.mean(maskInd[1])
            
            indices = np.nonzero(mei0)
            corner1 = max(indices[0])-maskCx
            corner2 = min(indices[0])-maskCx
            corner3 = max(indices[1])-maskCy
            corner4 = min(indices[1])-maskCy
            height = corner1 - corner2+1
            width = corner3 - corner4+1
            extend = meiSize/float(height*width)
        
        #features = [Hu1[0][0],Hu1[1][0],Hu1[2][0],Hu1[3][0],Hu1[4][0],Hu1[5][0],Hu1[6][0],
        #            Hu2[0][0],Hu2[1][0],Hu2[2][0],Hu2[3][0],Hu2[4][0],Hu2[5][0],Hu2[6][0],
        #            cx1, cy1, cx2, cy2, meiSize, corner1, corner2, corner3,corner4, height, width, extend, case]
      
        features = [Hu1[0][0],Hu1[1][0],Hu1[2][0],Hu1[3][0],Hu1[4][0],Hu1[5][0],Hu1[6][0],
                    Hu2[0][0],Hu2[1][0],Hu2[2][0],Hu2[3][0],Hu2[4][0],Hu2[5][0],Hu2[6][0],
                    cx1, cy1, cx2, cy2, corner1, corner2, corner3,corner4,
                    meiSize, height/(width+0.000001),height, width, extend, case]
        #zeroFeatures = [-200]*14
        #zeroFeatures = [0]*14
        #if case == 1:# Rest case
        #    featureWriter.writerow(features)
        #else:
        #    if features[0:len(features)-6] != zeroFeatures:
        #       featureWriter.writerow(features)
        featureWriter.writerow(features)
       
        prev_frame = frame.copy()
                            
        if DISPLAY:
            #draw_str(vis, (20, 20), visual_name)
            vis = cv2.cvtColor(vis0, cv2.COLOR_GRAY2BGR)
            mei = cv2.cvtColor(mei0, cv2.COLOR_GRAY2BGR)
            cv2.imshow('MHI', vis)
            #cv2.imshow('MEI', mei)
            cv2.imshow('Video',frame)
    
            if 0xff & cv2.waitKey(1) == 27:
                break
            
    cam.release()
    cv2.destroyAllWindows()


# In[ ]:

if __name__ == '__main__':
    import sys
    import glob
    from os import listdir
    
    activities = ['movement','rest','rotation']
    actLens = [45,11,33] # Full sizes
    
    folderRoot = 'C:\\PostDoctorProjects\\VideoEEGData\\Shank2_20150609\\Shank2TrainingClips' 
    leftValidRegionImage = 'C:\\PostDoctorProjects\\VideoEEGData\\Shank2_20150609\\Shan2_20150609_effect_region_top.tif'
    rightValidRegionImage = 'C:\\PostDoctorProjects\\VideoEEGData\\Shank2_20150609\\Shan2_20150609_effect_region_bottom.tif'
  
    maskRegionLeft = cv2.imread(leftValidRegionImage)
    maskRegionLeft = cv2.cvtColor(maskRegionLeft,cv2.COLOR_BGR2GRAY)
    ret,maskRegionLeft =  cv2.threshold(maskRegionLeft,80,1,cv2.THRESH_BINARY)
    
    maskRegionRight = cv2.imread(rightValidRegionImage)
    maskRegionRight = cv2.cvtColor(maskRegionRight,cv2.COLOR_BGR2GRAY)
    ret,maskRegionRight =  cv2.threshold(maskRegionRight,80,1,cv2.THRESH_BINARY)
      
    #cv2.imshow('validRegion',maskRegion)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    for case in range(0,3):
        print 'Feature exactracting: ' + activities[case]
        subFolderPath = folderRoot + '\\' + activities[case] + '\\' + '*.avi'
        
        videoNames = glob.glob(subFolderPath)#listdir(subFolderPath)
        featureSaveName = folderRoot + '\\' + activities[case] + '_features.csv'
        
        fout = open(featureSaveName, 'wb')
        featureWriter = csv.writer(fout,quoting=csv.QUOTE_NONE)
        for ii in range(0,actLens[case]):
            fullVideoName = videoNames[ii]
            print fullVideoName
            tmp1 = fullVideoName.split('\\')
            tmp2 = tmp1[-1].split('_')
            position = tmp2[0][-1]
            if position=='R':
                maskRegion = maskRegionRight
            elif position=='L':
                maskRegion = maskRegionLeft
            else:
                maskRegion = []
                print "rat position error!!"
            video_feature_extraction_save(fullVideoName, featureWriter, maskRegion,case, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,
                                          THRESH_VALUE,False)

        fout.close()
    
