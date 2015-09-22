# version 0 works for the MIT experimental set up
# added output video with results
  

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
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:,:,1] = 255
    
    
    fps = 15.0
    fourcc = cv2.cv.CV_FOURCC(*'XVID')

    outputVideoName = "shank2activityRecognitionResults.avi";
    VideoOutput = cv2.VideoWriter(outputVideoName,fourcc, fps, (w,h))   
    
    tag = medfilt(df['tag'],9)
    tag = medfilt(tag,9)
    df['SmoothTag'] = tag
    df.to_csv(featureCSVFileName,index = False)
    
    ii = 0        
    
    #cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ii)
     
    while (ii < video_len-1):
    #while (ii<1000):
        ii += 1
        ret, frame = cam.read()
        activity = activities[int(tag[ii])]  
                                   
        cv2.putText(frame, activity, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
        cv2.imshow('Video',frame)
        VideoOutput.write(frame)
        
        if 0xff & cv2.waitKey(1) == 27:
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    activities = ['movement','rest','rotation']
    
    featureCSVFileName = 'shank2_video_features.csv'
    df = pd.read_csv(featureCSVFileName)
   
    df.columns = ['Hu1[0][0]','Hu1[1][0]','Hu1[2][0]','Hu1[3][0]','Hu1[4][0]','Hu1[5][0]','Hu1[6][0]',
                    'Hu2[0][0]','Hu2[1][0]','Hu2[2][0]','Hu2[3][0]','Hu2[4][0]','Hu2[5][0]','Hu2[6][0]',
                    'cx1', 'cy1', 'cx2', 'cy2', 'corner1', 'corner2', 'corner3','corner4',
                    'meiSize', 'height/(width+0.000001)','height', 'width', 'extend', 'tag']
    
    testVideoName = 'C:\\PostDoctorProjects\\VideoEEGData\\Shank2_20150609\\video_stitched\\Shan2_20150609_cut.avi'    
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