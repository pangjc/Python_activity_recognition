# This script aims to extract and save features for each frame of a LONG video
# and manual scoring from a text file
#
# Read two manual scores
import numpy as np
import scipy
import cv2
from common import nothing, clock, draw_str
import csv
import os
import matplotlib.pyplot as plt
from pylab import ginput
import time

MHI_DURATION = 1
MAX_TIME_DELTA = 0.5
MIN_TIME_DELTA = 0.05
THRESH_VALUE = 60

def extract_background(videoName,startFrame):
    cap = cv2.VideoCapture(videoName)
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    ret,frame_it0 = cap.read()  
    frame_it = cv2.cvtColor(frame_it0, cv2.COLOR_BGR2GRAY)
    width = np.size(frame_it, 1) 
    height = np.size(frame_it, 0)
    sampleFrameNums = np.linspace(startFrame,length-10,200)
    sampleLen = len(sampleFrameNums)
    videoStack = np.zeros([height, width, sampleLen],dtype = 'uint8')
    mm = 0
    for jj in range(0,sampleLen):
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sampleFrameNums[jj])
        ret,frame_it0 = cap.read()
        frame_it = cv2.cvtColor(frame_it0, cv2.COLOR_BGR2GRAY)
        cv2.imshow('sample frame',frame_it)
        if 0xff & cv2.waitKey(2) == 27:
            break
        videoStack[:,:,mm] = frame_it
        mm+=1
    
    cap.release()  
    
    frame_ref0 = np.max(videoStack,axis = 2)
    #frame_ref0 = np.median(videoStack,axis = 2)
    frame_base = np.uint8(frame_ref0)
    return frame_base
 
def segmentation_frame(frame0,frame_base,maskRegion,kernel):
        
    frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(3,3),0)  
    diff = (frame-frame_base)*maskRegion
    
    #ret,fgmask0 = cv2.threshold(diff,70,255,cv2.THRESH_BINARY)
    ret,fgmask0 = cv2.threshold(diff,120,255,cv2.THRESH_BINARY)
    
    fgmask0 = fgmask0*maskRegion
    fgmask1 = cv2.morphologyEx(fgmask0,cv2.MORPH_OPEN,kernel)             
    fgmask2 = np.zeros_like(fgmask1,np.uint8)
    
    contour0, hier0 = cv2.findContours(fgmask1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    maxAreaSize = 0
    maxInd = 0
    
    # cv2.imshow('mask', fgmask0)
    
    cx, cy, mouse_size, orientation = 0, 0, 0, 0 
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
        Hu = cv2.HuMoments(M)
        smallNum = [1e-200]*7 
        
        Hu = np.sign(Hu)*np.log10(np.abs(Hu))

        
    mouse = np.zeros_like(frame0)
  
    mouse[:,:,0] = fgmask2;

            
    if contour0:
        mouse[cy-1:cy+1,cx-1:cx+1,1] = 255
      
    mouse_size = np.count_nonzero(fgmask2)  
            
    features =  [cx, cy, mouse_size, orientation,Hu[0][0],Hu[1][0],Hu[2][0],Hu[3][0],Hu[4][0],Hu[5][0],Hu[6][0]]
    
    return mouse,features 

def onChange(trackbarValue):
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
    ret,img = cap.read()
    cv2.imshow("Select starting frame", img)
    pass
 
def select_valid_region(frame_input):
    inputFig = plt.figure()
    plt.title('Select arm points')    
    plt.imshow(frame_input, cmap = 'gray', interpolation = 'bicubic')
    # Same as ginput() in Matlab 
    region_ptList = ginput(15)
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

    return mask0, mask

def feature_extraction_fullVideo(videoName, startFrame,maskRegion,featureWriter, videoOutput, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE,DISPLAY=False): 
    cv2.namedWindow('rat activity recognition')
    visuals = ['input', 'frame_diff', 'motion_hist', 'grad_orient']
    # use MHI features (motion history intensity)
    visual_name = visuals[2]
  
    cam = cv2.VideoCapture(videoName)
    video_len = cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame)
    
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:,:,1] = 255
    
    
    #fps = 15.0
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')

    #outputVideoName = "activityRecognitionResults.avi";
    #VideoOutput = cv2.VideoWriter(outputVideoName,fourcc, fps, (w,h))   
    
    frame_base = extract_background(videoName, startFrame)
    frame_base = cv2.GaussianBlur(frame_base,(3,3),0) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #cam.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ii)
     
    #for ii in range(int(startFrame)+1,int(video_len)-10):
    for ii in range(1,3000):
        ret, frame = cam.read()
        ## Mouse segmentation
        mouse, mFeat = segmentation_frame(frame,frame_base,mask,kernel)

        frame_results = cv2.add(frame,mouse)  
        
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
            
            vis0 = vis0*maskRegion
            mei0 = mei0*scipy.sign(maskRegion)

        elif visual_name == 'grad_orient':
            hsv[:,:,0] = mg_orient/2
            hsv[:,:,2] = mg_mask*255
            vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        ## Compute features
        M1 = cv2.moments(mei0)
        M2 = cv2.moments(vis0)    
        Hu1 = cv2.HuMoments(M1)
        Hu2 = cv2.HuMoments(M2)
        
        smallNum = [1e-200]*7 
        Hu1 = Hu1 + smallNum
        Hu2 = Hu2 + smallNum
        
        Hu1 = np.sign(Hu1)*np.log10(np.abs(Hu1))
        Hu2 = np.sign(Hu2)*np.log10(np.abs(Hu2))
      
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
        
        if meiSize == 0:
            corner1 = 0
            corner2 = 0
            corner3 = 0
            corner4 = 0
            height = 0
            width = 0
            extend = 0
        else:
            indices = np.nonzero(mei0)
            corner1 = max(indices[0])
            corner2 = min(indices[0])
            corner3 = max(indices[1])
            corner4 = min(indices[1])
            height = corner1 - corner2+1
            width = corner3 - corner4+1
            extend = meiSize/float(height*width)
            
        features = [ii,Hu1[0][0],Hu1[1][0],Hu1[2][0],Hu1[3][0],Hu1[4][0],Hu1[5][0],Hu1[6][0],
                    Hu2[0][0],Hu2[1][0],Hu2[2][0],Hu2[3][0],Hu2[4][0],Hu2[5][0],Hu2[6][0],
                    cx1, cy1, cx2, cy2, meiSize, corner1, corner2, corner3, corner4,height, width, extend,
                    mFeat[0],mFeat[1],mFeat[2],mFeat[3],mFeat[4],mFeat[5],mFeat[6],
                    mFeat[7],mFeat[8],mFeat[9],mFeat[10]]
       
        featureWriter.writerow(features)
         
        prev_frame = frame.copy()      
     
        vis = cv2.cvtColor(vis0, cv2.COLOR_GRAY2BGR)
        mei = cv2.cvtColor(mei0, cv2.COLOR_GRAY2BGR)
            
                            
   
        frameDist0 = np.concatenate((frame_results, vis), axis=0)
        frameDist = frameDist0[::2,::2,:]
        videoOutput.write(frameDist)   
    
        if DISPLAY:
            cv2.imshow('Video',frameDist)
            if 0xff & cv2.waitKey(1) == 27:
                break
    
         
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from sklearn.externals import joblib
    from os import listdir
    
    #videoName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2.avi'
    #videoName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam1_Cages3_4_combined\\JD_Cam1_2015_0514_Cages3_4.avi'
    #scoreTxtName1 = 'C:\\PostDoctorProjects\VideoEEGData\\20150514_JD_Cam1_Cages3_4_combined\\20150514_JD_mouse4_JP.txt' 
    #scoreTxtName2 = 'C:\\PostDoctorProjects\VideoEEGData\\20150514_JD_Cam1_Cages3_4_combined\\20150514_SMO_mouse4_JP.txt'      
    videoName = 'C:\\PostDoctorProjects\\VideoEEGData\\overnight827_828\\Box1_stitched\\Box1_2015_08_27_15_full.avi'
#-----------------------------------------------------
#
# Read video and get some information about it
#
#-----------------------------------------------------

(dirName,videoFileName) = os.path.split(videoName)
(videoBaseName, videoExtension) = os.path.splitext(videoFileName)

cap = cv2.VideoCapture(videoName)
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,200)
ret,frame_input0 = cap.read()      
frame_input = cv2.cvtColor(frame_input0,cv2.COLOR_BGR2GRAY)

width = np.size(frame_input, 1) 
height = np.size(frame_input, 0)
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

# print width, height, length, fps

#-----------------------------------------------------
#
# Specify the output video and the csv file
#
#-----------------------------------------------------
fourcc = cv2.cv.CV_FOURCC(*'XVID')

outputVideoName = dirName+"/"+videoBaseName+"_featuresComputation_full.avi";
videoOutput = cv2.VideoWriter(outputVideoName,fourcc, fps, (int(width/2),height))

csvOutputName = dirName+"/"+videoBaseName+"_features_full.csv"
#print(csvOutputName)
fout = open(csvOutputName, 'wb')
featureWriter = csv.writer(fout)
        
#-----------------------------------------------------
#
# Select the first frame to start segmentation
#
#-----------------------------------------------------

#cv2.namedWindow('Select the starting frame')
#cv2.createTrackbar( 'frame number', 'Select the starting frame', 0, length, onChange)
startFrame = 0
#while (1):  
#    startFrame = cv2.getTrackbarPos('frame number','Select the starting frame')
#    k = cv2.waitKey(1) & 0xFF
#    if k == 27:
#        break
    
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame+10000)
ret,frame_input0 = cap.read()      
frame_input = cv2.cvtColor(frame_input0,cv2.COLOR_BGR2GRAY)   
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame)
#-----------------------------------------------------
#
# Select the valid region and save the mask
#
#-----------------------------------------------------
'''
mask0, mask = select_valid_region(frame_input)
mask_toSave = np.zeros(mask0.shape[:3],dtype = 'uint8')
mask_toSave[:,:,2] = mask
mask_toSave[:,:,1] = mask
mask_toSave[:,:,0] = mask
validRegionImageName =  dirName+"/"+videoBaseName+"_effect_region.tif" 
cv2.imwrite(validRegionImageName,mask_toSave)
'''

validRegionImage = 'C:\\PostDoctorProjects\\VideoEEGData\\overnight827_828\\Box1_stitched\\Box1_2015_08_27_15_full_effect_region.tif'
#validRegionImage = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2_effect_region.tif'
#validRegionImage= 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam1_Cages3_4_combined\\JD_Cam1_2015_0514_Cages3_4_effect_region.tif'

maskRegion = cv2.imread(validRegionImage)
mask = cv2.cvtColor(maskRegion,cv2.COLOR_BGR2GRAY)


tic = time.time()
feature_extraction_fullVideo(videoName,startFrame, mask,featureWriter,videoOutput, MIN_TIME_DELTA,MAX_TIME_DELTA,MHI_DURATION,THRESH_VALUE,True)
toc = time.time()
print 'running time: ' + str(toc-tic)
fout.close()
videoOutput.release()