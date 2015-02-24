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
        #fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        fps = 15.0
        outputVideoName = dirName+"/"+videoBaseName+"_trackResults.avi";
        out = cv2.VideoWriter(outputVideoName,fourcc, fps, (width,height))
            
        #out = cv2.VideoWriter(outputVideoName,-1, fps, (width,height))
        # Same as ginput() in Matlab 
        region_ptList = ginput(8)
        plt.close(inputFig)
        
        region_ptArray0 = np.asarray(region_ptList)
        region_ptArray = region_ptArray0.astype(int)
        region_ptArray = np.vstack((region_ptArray, region_ptArray[0,:]))
        
        # Get the arms A,B,C,U
        pointsLeft = region_ptArray[0:4,:]
        maskLeft0 = np.zeros(frame_input.shape[:2],dtype = 'uint8')
        cv2.drawContours(maskLeft0,[pointsLeft],0,255,-1)
        maskLeft0 = cv2.cvtColor(maskLeft0,cv2.COLOR_GRAY2BGR)
        maskLeft = cv2.cvtColor(maskLeft0,cv2.COLOR_BGR2GRAY)
 
        pointsRight = region_ptArray[4:8,:]
        maskRight0 = np.zeros(frame_input.shape[:2],dtype = 'uint8')
        cv2.drawContours(maskRight0,[pointsRight],0,255,-1)
        maskRight0 = cv2.cvtColor(maskRight0,cv2.COLOR_GRAY2BGR)
        maskRight = cv2.cvtColor(maskRight0,cv2.COLOR_BGR2GRAY)
        
        mask0 = maskLeft0 + maskRight0
        mask = maskLeft + maskRight
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
            
            fgmask01 = fgmask0*maskLeft
            fgmask02 = fgmask0*maskRight
            
            fgmask11 = cv2.morphologyEx(fgmask01,cv2.MORPH_OPEN,kernel)   
            fgmask12 = cv2.morphologyEx(fgmask02,cv2.MORPH_OPEN,kernel)           
            fgmask21 = np.zeros_like(fgmask11,np.uint8)
            fgmask22 = np.zeros_like(fgmask12,np.uint8)
            
            # fgmask2 = img_as_ubyte(morphology.remove_small_objects(fgmask1>0, areaSize))
            # Only keep the component with the largest area size
            contour01, hier01 = cv2.findContours(fgmask11,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            maxAreaSize1 = 0
            maxInd1 = 0
            
            # Get the largest component and fill the holes by redrawing it
            if contour01:
                for jj in range(0,len(contour01)):
                    if cv2.contourArea(contour01[jj]) > maxAreaSize1:
                        maxAreaSize1 = cv2.contourArea(contour01[jj]) 
                        maxInd1 = jj  
                # Draw the largest component with holes filled                
                cv2.drawContours(fgmask21,[contour01[maxInd1]],0,255,-1)
                # Get the orientation        
                (junk11,junk12),(MA1,ma1),orientation1 = cv2.fitEllipse(contour01[maxInd1])
                # Get the centroid
                M1 = cv2.moments(contour01[maxInd1])
                cx1 = int(M1['m10']/M1['m00'])
                cy1 = int(M1['m01']/M1['m00'])
                     
            # Only keep the component with the largest area size
            contour02, hier02 = cv2.findContours(fgmask12,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            maxAreaSize2 = 0
            maxInd2 = 0
            
            # Get the largest component and fill the holes by redrawing it
            if contour02:
                for jj in range(0,len(contour02)):
                    if cv2.contourArea(contour02[jj]) > maxAreaSize2:
                        maxAreaSize2 = cv2.contourArea(contour02[jj]) 
                        maxInd2 = jj  
                # Draw the largest component with holes filled                
                cv2.drawContours(fgmask22,[contour02[maxInd2]],0,255,-1)
                # Get the orientation        
                (junk21,junk22),(MA2,ma2),orientation2 = cv2.fitEllipse(contour02[maxInd2])
                # Get the centroid
                M2 = cv2.moments(contour02[maxInd2])
                cx2 = int(M2['m10']/M2['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                
            
            mouse = np.zeros_like(frame0)

            mouse[:,:,1] = cv2.add(fgmask21,fgmask22)
            mouse[:,:,2] = cv2.add(fgmask21,fgmask22)
        
            if contour01:
                mouse[cy1-1:cy1+1,cx1-1:cx1+1,0] = 0
                mouse[cy1-1:cy1+1,cx1-1:cx1+1,1] = 0
            if contour01:
                mouse[cy2-1:cy2+1,cx2-1:cx2+1,0] = 0
                mouse[cy2-1:cy2+1,cx2-1:cx2+1,1] = 0
              
            frame_results = cv2.subtract(frame0,mouse) 
          
            cv2.imshow('frame0',frame_results)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            
            #fgmask2 = morphology.remove_small_objects(fgmask1,200)
            #fgmask = fgmask2.astype(int)
            out.write(frame_results)
        
            # Out put information such as frame number, area size etc
            ifr = round(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) 
            
            mouse_size1 = np.count_nonzero(fgmask21)
            mouse_size1 = np.count_nonzero(fgmask22)   
                 
            writer.writerow((videoName, ifr, cx1, cy1, mouse_size1, orientation1, cx1, cy1, mouse_size1, orientation1))
        
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
        
        
        
 
