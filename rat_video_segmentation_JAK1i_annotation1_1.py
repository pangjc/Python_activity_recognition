# version 0 works for the MIT experimental set up
# added output video with results
#
# Removed manual annotation from rat_video_segmentation_JAK1i_annotation
import numpy as np
import cv2
from common import nothing, clock, draw_str
import csv
import os
import pandas as pd
from scipy.signal import medfilt

MHI_DURATION = 1
MAX_TIME_DELTA = 0.5
MIN_TIME_DELTA = 0.05
THRESH_VALUE = 32

def video_results_annotation(videoName,featureCSVFileName, activities,df): 
    cam = cv2.VideoCapture(videoName)
    video_len = cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    ret, frame = cam.read()
    h, w = frame.shape[:2]

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:,:,1] = 255
    
    
    fps = 15.0
    fourcc = cv2.cv.CV_FOURCC(*'XVID')

    outputVideoName = "C:\\PostDoctorProjects\VideoEEGData\\20150520_Cam4_Subject_5+6_stitched\\JAK1i_RecognitionResults.avi";
    VideoOutput = cv2.VideoWriter(outputVideoName,fourcc, fps, (w,h))   
        
    ii = 0 
    nrow,ncol = df.shape       
    startFrame = int(df.loc[0,'iframe'])
    endFrame = int(df.loc[nrow-1,'iframe'])
    #cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ii)
    
    cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame)

    COM_score = np.array(df['predict'])
         
    annotationColor = [(255,0,0),(0,255,0),(0,0,255)]     
    for ii in range(endFrame-startFrame-1):
        ret, frame = cam.read()
        activityCOM = activities[int(COM_score[ii])]  
        
        cv2.putText(frame, 'COM: '+ activityCOM, (255, 85),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotationColor[int(COM_score[ii])])
        cv2.imshow('Video',frame)
        VideoOutput.write(frame)
        
        if 0xff & cv2.waitKey(1) == 27:
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    activities = ['0','1-3','4-6']
    
    featureCSVFileName = 'C:\\PostDoctorProjects\VideoEEGData\\20150520_Cam4_Subject_5+6_stitched\\20150520_Cam4_Subject_5+6_08-20-04_stitched_features_predicted.csv'
    df = pd.read_csv(featureCSVFileName,header = False)
       
    testVideoName = 'C:\\PostDoctorProjects\VideoEEGData\\20150520_Cam4_Subject_5+6_stitched\\20150520_Cam4_Subject_5+6_08-20-04_stitched.avi'    
    video_results_annotation(testVideoName,featureCSVFileName, activities,df)


'''
    for case in range(3,4):
        subFolderPath = folderRoot + '\\' + activities[case]
        
        videoNames = listdir(subFolderPath)
        actLens = [30,130,594,139,30,68,720,52]

        for ii in range(0,actLens[case]):
            testVideoName = subFolderPath +'\\'+ videoNames[ii]
            real_time_evaluation(testVideoName, featureWriter,classifier, activities, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE,True)
    fout.close() 
'''