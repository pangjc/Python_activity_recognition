import Tkinter as tk
import sys
import tkFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import ginput
import csv
import os.path

# A gui version for video segmentation 
# This script is customized from 'rat_video_processing_GUI_v2_0.py' for Meera Modi's experiment setup
# Works for the right and left cameras
# One bug exists: the output video with segmentation results is empty!!

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
 
    def client_load(self):
        videoName = tkFileDialog.askopenfilename()
        (dirName,videoBaseName0) = os.path.split(videoName)
        (videoBaseName, videoExtension) = os.path.splitext(videoBaseName0)
        
        ###########################
        ###########################
        ###########################   
             
        ###########################
        ###########################
        ###########################
        # Select points to detrmine Y maze arms
        cap = cv2.VideoCapture(videoName)
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,200)
        ret,frame_input0 = cap.read()      
        frame_input = cv2.cvtColor(frame_input0,cv2.COLOR_BGR2GRAY)
        cap.release()
        
        inputFig = plt.figure()
        plt.title('Select arm points')    
        plt.imshow(frame_input, cmap = 'gray', interpolation = 'bicubic')
        
        # Specify output video
        width = np.size(frame_input, 1) 
        height = np.size(frame_input, 0)
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        outputVideoName = dirName+"/"+videoBaseName+"_trackResults.avi";
        out = cv2.VideoWriter(outputVideoName,fourcc, fps, (width,height))
        
        if (not out.isOpened()):
            print 'error'
            
        #out = cv2.VideoWriter(outputVideoName,-1, fps, (width,height))
        # Same as ginput() in Matlab 
        region_ptList = ginput(8)
        plt.close(inputFig)
        
        region_ptArray0 = np.asarray(region_ptList)
        region_ptArray = region_ptArray0.astype(int)
        region_ptArray = np.vstack((region_ptArray, region_ptArray[0,:]))
        
        # Get the arms A,B,C,U
        mask_points = region_ptArray
        mask0 = np.zeros(frame_input.shape[:2],dtype = 'uint8')
        cv2.drawContours(mask0,[mask_points],0,255,-1)
        mask0 = cv2.cvtColor(mask0,cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(mask0,cv2.COLOR_BGR2GRAY)
 
        
        # Create color image of effect region to save 
        mask_toSave = np.zeros(mask0.shape[:3],dtype = 'uint8')
        mask_toSave[:,:,2] = mask
        mask_toSave[:,:,1] = mask
        mask_toSave[:,:,0] = mask
    
        
        YarmImageName =  dirName+"/"+videoBaseName+"_effect_region.tif" 
        cv2.imwrite(YarmImageName,mask_toSave)
        
        ###########################
        ###########################
        ###########################
        
        ###########################
        ###########################
        ###########################
        # Read the whole video and extract the background as the base frame instead of using a slider
        cap = cv2.VideoCapture(videoName)
        length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
  
        ret,frame_it0 = cap.read()  
        frame_it = cv2.cvtColor(frame_it0, cv2.COLOR_BGR2GRAY)
        width = np.size(frame_it, 1) 
        height = np.size(frame_it, 0)
        sampleFrameNums = np.linspace(10,length-10,200)
        sampleLen = len(sampleFrameNums)
        videoStack = np.zeros([height, width, sampleLen],dtype = 'uint8')
        mm = 0
        for jj in range(0,sampleLen):
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sampleFrameNums[jj])
            ret,frame_it0 = cap.read()
            frame_it = cv2.cvtColor(frame_it0, cv2.COLOR_BGR2GRAY)
            cv2.imshow('sample frame',frame_it)
            if 0xff & cv2.waitKey(10) == 27:
                break
            videoStack[:,:,mm] = frame_it
            mm+=1
        
        cap.release()  
        
        frame_ref0 = np.min(videoStack,axis = 2)
        frame_base = np.uint8(frame_ref0)
        
        backgroundImageName =  dirName+"/"+videoBaseName+"_background.tif" 
        cv2.imwrite(backgroundImageName,frame_base)
        
        ################################
        ################################
        
        # Smooth the reference/base frame        
        frame_base = cv2.GaussianBlur(frame_base,(7,7),0) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        
        # Set output the csv file 
        csvOutputName = dirName+"/"+videoBaseName+"_features.csv"
        #print(csvOutputName)
        fout = open(csvOutputName, 'wb')
        writer = csv.writer(fout)
        ###writer.writerow( ('video name','frame number', 'centroid x', 'centroid y','size','orientation','proportion A','proportion B','proportion C','proportion U') )

        # Main loop for video processing
        #while(cap.isOpened()):
        cap = cv2.VideoCapture(videoName)
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,50)
        for ii in range(50,length-10):
            ret,frame0 = cap.read()
            
            frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame,(7,7),0)  
            diff = (frame-frame_base)*mask
              
            ret,fgmask0 = cv2.threshold(diff,150,255,cv2.THRESH_BINARY_INV)
            
            fgmask0 = fgmask0*mask
            
            fgmask1 = cv2.morphologyEx(fgmask0,cv2.MORPH_OPEN,kernel)            
            fgmask2 = np.zeros_like(fgmask1,np.uint8)
            
            # fgmask2 = img_as_ubyte(morphology.remove_small_objects(fgmask1>0, areaSize))
            # Only keep the component with the largest area size
            contour0, hier0 = cv2.findContours(fgmask1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            maxAreaSize = 0
            maxInd = 0
            
            # Get the largest component and fill the holes by redrawing it
            if contour0:
                for jj in range(0,len(contour0)):
                    if cv2.contourArea(contour0[jj]) > maxAreaSize:
                        maxAreaSize = cv2.contourArea(contour0[jj]) 
                        maxInd = jj  
                # Draw the largest component with holes filled                
                cv2.drawContours(fgmask2,[contour0[maxInd]],0,255,-1)
                # Get the orientation        
                (junk1,junk2),(MA,ma),orientation = cv2.fitEllipse(contour0[maxInd])
                # Get the centroid
                M = cv2.moments(contour0[maxInd])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                     
            mouse = np.zeros_like(frame0)
            mouse[:,:,0] = fgmask2;
        
            if contour0:
                mouse[cy-1:cy+1,cx-1:cx+1,1] = 255
              
            frame_results = cv2.add(frame0,mouse) 
          
            cv2.imshow('frame0',frame_results)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            
            #fgmask2 = morphology.remove_small_objects(fgmask1,200)
            #fgmask = fgmask2.astype(int)
            out.write(frame_results)
        
            # Out put information such as frame number, area size etc
            ifr = round(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) 
            mouse_size  = np.count_nonzero(mouse)
                 
            writer.writerow((videoName, ifr, cx, cy, mouse_size, orientation))
        
        cap.release()
        out.release()
        fout.close()
        cv2.destroyAllWindows()
    
    def client_exit(self):
        sys.exit(0)
        
        
    
root = tk.Tk()
root.geometry("300x100")
app = Window(root)

root.mainloop()
        
        
        
 