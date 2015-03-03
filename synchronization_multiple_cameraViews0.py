import Tkinter as tk
import sys
import tkFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import ginput
import csv
import os.path
from cv2 import VideoCapture

# Trying to investigate the synchronization problems

class Window(tk.Frame):
    
    def __init__(self, master = None):
        tk.Frame.__init__(self,master)
        self.master = master
        self.init_window()
        
    def init_window(self):        
        self.master.title("Rat video segmentation")
        self.pack(fill = tk.BOTH, expand = 1)
        loadVideoButton = tk.Button(self,text = "Load Video", command = self.client_load)
        quitButton = tk.Button(self,text = "Quit", command = self.client_exit)
        loadVideoButton.place(x = 0, y = 0)
        quitButton.place(x = 80, y = 0)       
 
    def client_exit(self):
        sys.exit(0)
        
 
    def client_load(self):
         
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
        
        # Specify videos to be analyzed
        videoNameSide1 = tkFileDialog.askopenfilename()
        (dirNameSide1,videoBaseNameSide10) = os.path.split(videoNameSide1)
        (videoBaseNameSide1, videoExtensionSide1) = os.path.splitext(videoBaseNameSide10)
        
        videoNameTop = tkFileDialog.askopenfilename()
        (dirNameTop,videoBaseNameTop0) = os.path.split(videoNameTop)
        (videoBaseNameTop, videoExtensionTop) = os.path.splitext(videoBaseNameTop0) 
       
        frame_inputSide1, widthSide1, heightSide1,fpsSide1 = get_video_information(videoNameSide1)
        frame_inputTop, widthTop, heightTop,fpsTop = get_video_information(videoNameTop)   
        
        
        # Main loop for video processing
        capSide1 = cv2.VideoCapture(videoNameSide1)
        capSide1.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,50)
        lengthSide1 = int(capSide1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
        capTop = cv2.VideoCapture(videoNameTop)
        capTop.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,50)
        lengthTop = int(capTop.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        
        timeStampPrevSide1 = capSide1.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        timeStampPrevTop = capTop.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            
        length = min([lengthSide1,lengthTop])
        for ii in range(50,length-10):
            ret,frameSide10 = capSide1.read()
            ret,frameTop0 = capTop.read()
            
            fpsSide1 = capSide1.get(cv2.cv.CV_CAP_PROP_FPS)
            fpsTop = capTop.get(cv2.cv.CV_CAP_PROP_FPS)
            #print 'side camera time: ' + str(capSide1.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            #print 'top camera time: ' + str(capTop.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            #timeStampDiff = capTop.get(cv2.cv.CV_CAP_PROP_POS_MSEC)-capSide1.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            #print 'Time difference: ' + str(timeStampDiff)
            timeStampSide1 = capSide1.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            timeStampTop = capTop.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            
            timeStampDiffSide1 = timeStampSide1-timeStampPrevSide1 
            timeStampDiffTop = timeStampTop-timeStampPrevTop  
            
            timeStampPrevSide1 = timeStampSide1
            timeStampPrevTop = timeStampTop
               
            print 'Channel time diff: ' + str(timeStampSide1-timeStampTop) + ' side frame diff: ' + str(timeStampDiffSide1) + ' top frame diff: ' + str(timeStampDiffTop) + ' side frame frame: ' + str(fpsSide1) + 'top frame rate: ' + str(fpsTop)
                  
            cv2.imshow('Top View',frameTop0)
            cv2.imshow('Side View',frameSide10)
            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break
            
 
        
        capSide1.release()

        capTop.release()
  
        cv2.destroyAllWindows()
    
root = tk.Tk()
root.geometry("300x100")
app = Window(root)

root.mainloop()
        
        
        
 
