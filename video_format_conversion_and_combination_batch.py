import Tkinter as tk
import os,sys
import tkFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import ginput
import csv
import os.path
from cv2 import VideoCapture
from os import listdir

# The Y-Cam uses 32 bit codec which can not be opened by matlab
# This script does nothing just read, multiple videos and write ONE combined video                

# This script is based on video_format_conversion_and_combination.py
# The modification is for batch processing for a number of folders containing video segments 
# 
def get_video_information(videoName):
    # Load the video and get one frame to display and video information
    cap = cv2.VideoCapture(videoName)
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,200)
    ret,frame_input0 = cap.read()      
    frame_input = cv2.cvtColor(frame_input0,cv2.COLOR_BGR2GRAY)               
    width = np.size(frame_input, 1) 
    height = np.size(frame_input, 0)
    #fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    fps = 15.0
    cap.release()  
    return frame_input, width, height, fps
    # end of function definition: get_video_information

def get_frame_numbers(videoNameRoot,videoNames,startVideoInd,videoNum):
    frameLens = []
    for ii in range(startVideoInd,startVideoInd+videoNum):
        videoFullName = videoNameRoot + '\\' + videoNames[ii]
        capSeg1 = cv2.VideoCapture(videoFullName)
        lengthSeg1 = int(capSeg1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frameLens.append(lengthSeg1) 
       
    return frameLens        

# Specify the ROOT Path of containing subfolders having all video segments 
RootPath = 'C:\\PostDoctorProjects\\VideoEEGData\JAK1i20150421_raw\\LU'
OutPutRootPath = RootPath
if not os.path.exists(OutPutRootPath):
    os.makedirs(OutPutRootPath)
  
subdirectories = os.listdir(RootPath)
   
folderNum = len(subdirectories)

for ii in xrange(folderNum):
    videoNameRoot = RootPath+'\\'+subdirectories[ii]
    videoNames = listdir(videoNameRoot)
    videoNum = len(videoNames)
    videoNameSeg1 = videoNameRoot+'\\' + videoNames[0]
    
    (dirNameSeg1,videoBaseNameSeg10) = os.path.split(videoNameSeg1)   
    (videoBaseNameSeg1, videoExtensionSeg1) = os.path.splitext(videoBaseNameSeg10)
    
    dirNameSegOut  = videoNameRoot+'_stitched'
    if not os.path.exists(dirNameSegOut ):
        os.makedirs(dirNameSegOut )


    frame_input, width, height,fps = get_video_information(videoNameSeg1)
    #frame_inputTop, widthTop, heightTop,fpsTop = get_video_information(videoNameTop)   

    # Specify output videos
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    outputVideoNameSeg = dirNameSegOut+"/"+videoBaseNameSeg1+".avi";
    outSeg = cv2.VideoWriter(outputVideoNameSeg,fourcc, fps, (width,height)) 

    # Main loop for video processing
    videoLens = get_frame_numbers(videoNameRoot,videoNames,0,videoNum)


    print videoLens
    print sum(videoLens)

    for jj in range(0,videoNum):
        print str(jj+1) + ' out of ' + str(videoNum)
        videoFullName = videoNameRoot + '\\' + videoNames[jj]
        capSeg = cv2.VideoCapture(videoFullName)
        
        for kk in range(0,videoLens[jj]):
            ret,frameSeg10 = capSeg.read()
        
            #    cv2.imshow('Side View',frameSeg10)
            #    k = cv2.waitKey(1) & 0xff
            #    if k == 27:
            #        break
        
            outSeg.write(frameSeg10)
    
    
    outSeg.release()

capSeg1 = cv2.VideoCapture(outputVideoNameSeg)
lengthOutput = int(capSeg1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))     
print lengthOutput        
        