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

# This scrit aims to cut pieces from a long video by manually picking starting
# and ending frame numbers
class Window(tk.Frame):
    
    def __init__(self, master = None):
        tk.Frame.__init__(self,master)
        self.master = master
        self.init_window()
        
    def init_window(self):        
        self.master.title("Rat video cut")
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
        videoNameSeg1 = tkFileDialog.askopenfilename()
        (dirNameSeg1,videoBaseNameSeg10) = os.path.split(videoNameSeg1)
        (videoBaseNameSeg1, videoExtensionSeg1) = os.path.splitext(videoBaseNameSeg10)
        
       
        # specify start frame numbers and length of the piece
        
        startFrame = 76676  
        length = 8995
        frame_input, width, height,fps = get_video_information(videoNameSeg1)
        #frame_inputTop, widthTop, heightTop,fpsTop = get_video_information(videoNameTop)   
        
        # Specify output videos
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        outputVideoNameSeg = dirNameSeg1+"/"+videoBaseNameSeg1 + "_" + str(startFrame) +".avi";
        outSeg = cv2.VideoWriter(outputVideoNameSeg,fourcc, fps, (width,height)) 
        
        # Main loop for video processing
        capSeg1 = cv2.VideoCapture(videoNameSeg1)
        #lengthSeg1 = int(capSeg1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        #length = lengthSeg1
        capSeg1.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame)
        for ii in range(0,length):
            ret,frameSeg10 = capSeg1.read()
    
        #    cv2.imshow('Side View',frameSeg10)
        #    k = cv2.waitKey(1) & 0xff
        #    if k == 27:
        #        break
      
            outSeg.write(frameSeg10)


        outSeg.release()
        print "cut finished!"        
       ######capTop.release()
       ######cv2.destroyAllWindows()
    
root = tk.Tk()
root.geometry("300x100")
app = Window(root)

root.mainloop()
        
        
        
 