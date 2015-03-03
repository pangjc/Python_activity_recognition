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

# A gui version for video segmentation for both side views and top views simultaneously 
# Customzied and converted from rat_video_segmentation_sideView_GUI1.py and rat_video_segmentation_topView_GUI1.py

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
        
        def segmentation_frame(frame0,frame_base,maskRegion,kernel):
                
            frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame,(7,7),0)  
            diff = (frame-frame_base)*maskRegion
            
            ret,fgmask0 = cv2.threshold(diff,150,255,cv2.THRESH_BINARY_INV)
            
            fgmask0 = fgmask0*maskRegion
            fgmask1 = cv2.morphologyEx(fgmask0,cv2.MORPH_OPEN,kernel)             
            fgmask2 = np.zeros_like(fgmask1,np.uint8)
            
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
                (junk1,junk2),(MA,ma),orientation= cv2.fitEllipse(contour0[maxInd])
                # Get the centroid
                M = cv2.moments(contour0[maxInd])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
    
            mouse = np.zeros_like(frame0)
    
            mouse[:,:,1] = fgmask2
            mouse[:,:,2] = fgmask2
          
            if contour0:
                mouse[cy-1:cy+1,cx-1:cx+1,0] = 0
                mouse[cy-1:cy+1,cx-1:cx+1,1] = 0
              
            mouse_size = np.count_nonzero(fgmask2)  
            features =  [cx, cy, mouse_size, orientation]
            
            return mouse,features
        # end of function definition: segmentation_frame
        def extract_background(videoName):
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
            return frame_base
        # end of function definition: extract_background
        def select_valid_region_topView(frame_input):
            inputFig = plt.figure()
            plt.title('Select points to define valid regions:top View')    
            plt.imshow(frame_input, cmap = 'gray', interpolation = 'bicubic')
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
            mask_toSave = np.zeros(mask0.shape[:3],dtype = 'uint8')
            mask_toSave[:,:,2] = mask
            mask_toSave[:,:,1] = mask
            mask_toSave[:,:,0] = mask
        
            return maskLeft0, maskLeft, maskRight0, maskRight, mask_toSave

            # end of function definition: select_valid_region_topView
        
        
        # Small GUI to load video to be processed
        def select_valid_region_sideView(frame_input):
            inputFig = plt.figure()
            plt.title('Select points to define valid regions:side View')    
            plt.imshow(frame_input, cmap = 'gray', interpolation = 'bicubic')
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
        
            mask_toSave = np.zeros(mask0.shape[:3],dtype = 'uint8')
            mask_toSave[:,:,2] = mask
            mask_toSave[:,:,1] = mask
            mask_toSave[:,:,0] = mask
            return mask0, mask, mask_toSave
            # end of function definition:select_valid_region_sideView
         
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
        
        # Specify output videos
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        outputVideoNameSide1 = dirNameSide1+"/"+videoBaseNameSide1+"_trackResults.avi";
        outSide1 = cv2.VideoWriter(outputVideoNameSide1,fourcc, fpsSide1, (widthSide1,heightSide1)) 
        outputVideoNameTop = dirNameTop+"/"+videoBaseNameTop+"_trackResults.avi";
        outTop = cv2.VideoWriter(outputVideoNameTop,fourcc, fpsTop, (widthTop,heightTop)) 
        
        # Get valid region and save it
        maskSide10, maskSide1, maskToSaveSide1 = select_valid_region_sideView(frame_inputSide1)
        maskTopLeft0, maskTopLeft, maskTopRight0, maskTopRight, maskToSaveTop = select_valid_region_topView(frame_inputTop)
        validRegionImageNameSide1 =  dirNameSide1+"/"+videoBaseNameSide1+"_effect_region.tif" 
        cv2.imwrite(validRegionImageNameSide1,maskToSaveSide1)    
        validRegionImageNameTop =  dirNameTop+"/"+videoBaseNameTop+"_effect_region.tif" 
        cv2.imwrite(validRegionImageNameTop,maskToSaveTop)  
        
        # Read the whole video and extract the background as the base frame instead of using a slider
        frame_baseSide1 = extract_background(videoNameSide1)
        backgroundImageNameSide1 =  dirNameSide1 + "/"+videoBaseNameSide1 +"_background.tif" 
        cv2.imwrite(backgroundImageNameSide1,frame_baseSide1)     
        
        frame_baseTop = extract_background(videoNameTop)
        backgroundImageNameTop =  dirNameTop + "/"+videoBaseNameTop +"_background.tif" 
        cv2.imwrite(backgroundImageNameTop,frame_baseTop)   
        
        # Smooth the reference/base frame        
        frame_baseSide1 = cv2.GaussianBlur(frame_baseSide1,(7,7),0) 
        frame_baseTop = cv2.GaussianBlur(frame_baseTop,(7,7),0) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        
        # Set output the csv file 
        csvOutputNameSide1 = dirNameSide1 + "/" + videoBaseNameSide1 + "_features.csv"
        csvOutputNameTop = dirNameTop + "/" + videoBaseNameTop + "_features.csv"
        #print(csvOutputName)
        foutSide1 = open(csvOutputNameSide1, 'wb')
        writerSide1 = csv.writer(foutSide1)
        foutTop = open(csvOutputNameTop, 'wb')
        writerTop = csv.writer(foutTop)
        
        # Main loop for video processing
        capSide1 = cv2.VideoCapture(videoNameSide1)
        capSide1.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,50)
        lengthSide1 = int(capSide1.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
        capTop = cv2.VideoCapture(videoNameTop)
        capTop.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,50)
        lengthTop = int(capTop.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        
        length = min([lengthSide1,lengthTop])
        for ii in range(50,length-10):
            ret,frameSide10 = capSide1.read()
            ret,frameTop0 = capTop.read()
            
            #print 'side camera time: ' + str(capSide1.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            #print 'top camera time: ' + str(capTop.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
            # Top view camera    
            mouseTopLeft, featuresTopLeft = segmentation_frame(frameTop0,frame_baseTop,maskTopLeft,kernel)
            mouseTopRight, featuresTopRight = segmentation_frame(frameTop0,frame_baseTop,maskTopRight,kernel)
            
            mouseTop = np.zeros_like(frameTop0)  
            mouseTop = cv2.add(mouseTopLeft,mouseTopRight)
            frame_resultsTop = cv2.subtract(frameTop0,mouseTop) 
            
            # side view camera
            mouseSide1, featuresSide1 = segmentation_frame(frameSide10,frame_baseSide1,maskSide1,kernel)
        
            frame_resultsSide1 = cv2.subtract(frameSide10,mouseSide1) 
                 
            cv2.imshow('Top View',frame_resultsTop)
            cv2.imshow('Side View',frame_resultsSide1)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            
            #fgmask2 = morphology.remove_small_objects(fgmask1,200)
            #fgmask = fgmask2.astype(int)
            outSide1.write(frame_resultsSide1)
            outTop.write(frame_resultsTop)
            # Out put information such as frame number, area size etc
            ifrSide1 = round(capSide1.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) 
            featuresSide1 = [videoNameSide1] + [ifrSide1] + featuresSide1   
            writerSide1.writerow(featuresSide1)
                        
            ifrTop = round(capTop.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) 
            featuresTop = [videoNameSide1] + [ifrTop] + featuresTopLeft + featuresTopRight   
            writerTop.writerow(featuresTop)
        
        capSide1.release()
        outSide1.release()
        foutSide1.close()
        
        capTop.release()
        outTop.release()
        foutTop.close()
        cv2.destroyAllWindows()
    
root = tk.Tk()
root.geometry("300x100")
app = Window(root)

root.mainloop()
        
        
        
 
