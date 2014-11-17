#!/usr/bin/env python

import numpy as np
import cv2
from common import nothing, clock, draw_str

MHI_DURATION = 0.5
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05
THRESH_VALUE = 32

def draw_motion_comp(vis, (x, y, w, h), angle, color):
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0))
    r = min(w/2, h/2)
    cx, cy = x+w/2, y+h/2
    angle = angle*np.pi/180
    cv2.circle(vis, (cx, cy), r, color, 3)
    cv2.line(vis, (cx, cy), (int(cx+np.cos(angle)*r), int(cy+np.sin(angle)*r)), color, 3)

def mhi_video_extraction(videoName, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE): 
    cv2.namedWindow('rat activity recognition')
    visuals = ['input', 'frame_diff', 'motion_hist', 'grad_orient']
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
        mg_mask, mg_orient = cv2.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5 )
        seg_mask, seg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)
        ##### Debug visualizaton
      

        #####
        if visual_name == 'input':
            vis = frame.copy()
        elif visual_name == 'frame_diff':
            vis = frame_diff.copy()
        elif visual_name == 'motion_hist':
            vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif visual_name == 'grad_orient':
            hsv[:,:,0] = mg_orient/2
            hsv[:,:,2] = mg_mask*255
            vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        draw_str(vis, (20, 20), visual_name)
        cv2.imshow('MHI', vis)
        cv2.imshow('Video',frame)

        prev_frame = frame.copy()
        if 0xff & cv2.waitKey(50) == 27:
            break

            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    from os import listdir
    
    case = 7
    
    activities = ['drink','eat','groom','hang','head','rear','rest','walk']
    
    folderRoot = 'C:\\InternProjects\\rat_activity_recognition\\MIT_Traning_samples' 
    subFolderPath = folderRoot + '\\' + activities[case]
        
    videoNames = listdir(subFolderPath)

    for ii in range(0,20):
        fullVideoName = subFolderPath +'\\'+ videoNames[ii]
        mhi_video_extraction(fullVideoName, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE) 

    