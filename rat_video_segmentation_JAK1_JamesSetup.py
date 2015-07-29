import numpy as np
import cv2
import csv
import os.path
import matplotlib.pyplot as plt
from pylab import ginput


help_message = '''
USAGE: behavior features study for JAKli videos [<video_source>]

'''
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
    region_ptList = ginput(11)
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
    ret,fgmask0 = cv2.threshold(diff,80,255,cv2.THRESH_BINARY)
    
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

    mouse = np.zeros_like(frame0)
  
    mouse[:,:,0] = fgmask2;

            
    if contour0:
        mouse[cy-1:cy+1,cx-1:cx+1,1] = 255
      
    mouse_size = np.count_nonzero(fgmask2)  
    features =  [cx, cy, mouse_size, orientation]
    
    return mouse,features 
        
if __name__ == '__main__':
    import sys
    try:
        videoName = sys.argv[1]
    except:
        #fn = 0
        videoName = 'C:\\PostDoctorProjects\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2.avi'
        print help_message


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

outputVideoName = dirName+"/"+videoBaseName+"_trackResults.avi";
out = cv2.VideoWriter(outputVideoName,fourcc, fps, (width,height))

csvOutputName = dirName+"/"+videoBaseName+"_features.csv"
#print(csvOutputName)
fout = open(csvOutputName, 'wb')
writer = csv.writer(fout)
        
#-----------------------------------------------------
#
# Select the first frame to start segmentation
#
#-----------------------------------------------------

cv2.namedWindow('Select the starting frame')
cv2.createTrackbar( 'frame number', 'Select the starting frame', 0, length, onChange)
startFrame = 0
while (1):  
    startFrame = cv2.getTrackbarPos('frame number','Select the starting frame')
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,startFrame)
ret,frame_input0 = cap.read()      
frame_input = cv2.cvtColor(frame_input0,cv2.COLOR_BGR2GRAY)   

#-----------------------------------------------------
#
# Select the valid region and save the mask
#
#-----------------------------------------------------

mask0, mask = select_valid_region(frame_input)
mask_toSave = np.zeros(mask0.shape[:3],dtype = 'uint8')
mask_toSave[:,:,2] = mask
mask_toSave[:,:,1] = mask
mask_toSave[:,:,0] = mask


validRegionImageName =  dirName+"/"+videoBaseName+"_effect_region.tif" 
cv2.imwrite(validRegionImageName,mask_toSave)

#-----------------------------------------------------
#
# Extract the background and save it
#
#-----------------------------------------------------

frame_base = extract_background(videoName, startFrame)

backgroundImageName =  dirName+"/"+videoBaseName+"_background.tif" 
cv2.imwrite(backgroundImageName,frame_base)


 # Main loop for video processing
        #while(cap.isOpened()):
frame_base = cv2.GaussianBlur(frame_base,(3,3),0) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
for ii in range(0,length-5):
    ret,frame0 = cap.read()
    
    mouse, features = segmentation_frame(frame0,frame_base,mask,kernel)

    #frame_results = cv2.subtract(frame0,mouse) 
    frame_results = cv2.add(frame0,mouse)  
    
    if False:     
        cv2.imshow('frame0',frame_results)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    #fgmask2 = morphology.remove_small_objects(fgmask1,200)
    #fgmask = fgmask2.astype(int)
    out.write(frame_results)

    # Out put information such as frame number, area size etc
    ifr = round(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) 
    
    features = [videoName] + [ifr] + features
    writer.writerow(features)

cap.release()
out.release()
fout.close()
cv2.destroyAllWindows()