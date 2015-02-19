#!/usr/bin/env python

# Added generation of csv files containing features
# Added loop for all activities
import numpy as np
import cv2
from common import nothing, clock, draw_str
import csv
import os


MHI_DURATION = 15
MAX_TIME_DELTA = 10
MIN_TIME_DELTA = 5
THRESH_VALUE = 32

# This function aims to extract features for various activities 
def video_feature_extraction_save(videoName, featureWriter, case, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE,DISPLAY=False): 
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
    while (ii<video_len):
        ii += 1
        ret, frame = cam.read()
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
     
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
        features = [Hu1[0][0],Hu1[1][0],Hu1[2][0],Hu1[3][0],Hu1[4][0],Hu1[5][0],Hu1[6][0],
                                Hu2[0][0],Hu2[1][0],Hu2[2][0],Hu2[3][0],Hu2[4][0],Hu2[5][0],Hu2[6][0],
                                cx1, cy1, cx2, cy2, meiSize, case]
      
        zeroFeatures = [0]*14
        if case == 6:# Rest case
            featureWriter.writerow(features)
        else:
            if features[0:len(features)-6] != zeroFeatures:
                featureWriter.writerow(features)
       
        prev_frame = frame.copy()
                            
        if DISPLAY:
            #draw_str(vis, (20, 20), visual_name)
            vis = cv2.cvtColor(vis0, cv2.COLOR_GRAY2BGR)
            mei = cv2.cvtColor(mei0, cv2.COLOR_GRAY2BGR)
            cv2.imshow('MHI', vis)
            cv2.imshow('MEI', mei)
            cv2.imshow('Video',frame)
    
            if 0xff & cv2.waitKey(50) == 27:
                break
            
    cam.release()
    cv2.destroyAllWindows()

        
if __name__ == '__main__':
    import sys
    from os import listdir
    
    activities = ['drink','eat','groom','hang','head','rear','rest','walk']
    #actLens = [30,130,594,139,30,68,720,52] # Full sizes
    actLens = [30,120,200,120,30,60,300,50]
    
    folderRoot = 'C:\\InternProjects\\rat_activity_recognition\\MIT_Traning_samples' 

    
    for case in range(0,8):
        print 'Feature exactracting: ' + activities[case]
        subFolderPath = folderRoot + '\\' + activities[case]
        
        videoNames = listdir(subFolderPath)
        featureSaveName = folderRoot + '\\' + '_Feature_Exact' + '\\' + activities[case] + '_features.csv'
        
        fout = open(featureSaveName, 'wb')
        featureWriter = csv.writer(fout,quoting=csv.QUOTE_NONE)
        for ii in range(0,actLens[case]):
            fullVideoName = subFolderPath +'\\'+ videoNames[ii]
     
            #mhi_video_extraction(fullVideoName, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE) 
            video_feature_extraction_save(fullVideoName, featureWriter, case, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE,False)

        fout.close()
        # Set output the csv file 
        #csvOutputName = dirName+"/"+videoBaseName+"_features.csv"
        ##print(csvOutputName)
        #fout = open(csvOutputName, 'wb')
        #writer = csv.writer(fout)
        #writer.writerow( ('video name','frame number', 'centroid x', 'centroid y','size','orientation','proportion A','proportion B','proportion C','proportion U') )

    