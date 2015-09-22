# Adding moving velocity to the features 
import numpy as np
import cv2
import csv
import os.path
import time
import matplotlib.pyplot as plt
from pylab import ginput

def select_valid_region(frame_input):
    inputFig = plt.figure()
    plt.title('Select valid region')    
    plt.imshow(frame_input, cmap = 'gray', interpolation = 'bicubic')
    # Same as ginput() in Matlab 
    region_ptList = ginput(10)
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

help_message = '''
USAGE: D1.py [<video_source>]

'''


if __name__ == '__main__':
    import sys
    try:
        videoName = sys.argv[1]
    except:
        #fn = 0
        #videoName = 'C:\\PostDoctorProjects\\Python_activity_rotation\\MP_2014-12-19_10-47-29_600.mov'
        videoName = 'C:\\PostDoctorProjects\\VideoEEGData\\Shank2_20150609\\video_stitched\\Shan2_20150609.avi'
        print help_message

#-----------------------------------------------------
#
# compute background
#
#-----------------------------------------------------

(dirName,videoFileName) = os.path.split(videoName)
(videoBaseName, videoExtension) = os.path.splitext(videoFileName)

csvOut   = dirName+"/"+videoBaseName+"_out.csv"
fout = open(csvOut, 'wb')
csvWriter = csv.writer(fout)
csvWriter.writerow(['videoBaseName','ifr','cnt_area','Cx','Cy','cnt_solidity','Eccentricity','Angle'])


cap = cv2.VideoCapture(videoName)
# fps = cap.get(cv2.CAP_PROP_FPS)
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))  
  
ret,frame_it0 = cap.read()  
frame_it = cv2.cvtColor(frame_it0, cv2.COLOR_BGR2GRAY)
width = np.size(frame_it, 1) 
height = np.size(frame_it, 0)
print width,height


# Specify output video
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
 
videoOut = dirName+"/"+videoBaseName+"_out_top.avi"
videoOutWriter = cv2.VideoWriter(videoOut,fourcc, 15, (width,height))

print videoOut, fps, width,height
        
sampleFrameNums = np.linspace(10,length-10,100)
sampleLen = len(sampleFrameNums)
videoStack = np.zeros([height, width, sampleLen],dtype = 'uint8')
mm = 0
for jj in range(0,sampleLen):
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sampleFrameNums[jj])
    ret,frame_it0 = cap.read()
    frame_it = cv2.cvtColor(frame_it0, cv2.COLOR_BGR2GRAY)
    cv2.imshow('sample frame',frame_it)
    if 0xff & cv2.waitKey(5) == 27:
        break
    videoStack[:,:,mm] = frame_it
    mm+=1
cap.release()  

frame_ref0 = np.median(videoStack,axis = 2)
#frame_ref0 = np.amin(videoStack,axis = 2)
frame_base = np.uint8(frame_ref0)

backgroundImageName =  "./background.tif" 
cv2.imwrite(backgroundImageName,frame_base)

#-----------------------------------------------------
#
# Select the valid region and save the mask
#
#-----------------------------------------------------

mask0, mask = select_valid_region(frame_base)
mask_toSave = np.zeros(mask0.shape[:3],dtype = 'uint8')
mask_toSave[:,:,2] = mask
mask_toSave[:,:,1] = mask
mask_toSave[:,:,0] = mask


validRegionImageName =  dirName+"/"+videoBaseName+"_effect_region.tif" 
cv2.imwrite(validRegionImageName,mask_toSave)
#cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
cap = cv2.VideoCapture(videoName)
tic = time.time()
for ii in range(0,length):
    ret,frame0 = cap.read()
    
    frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(7,7),0)  
    #diff = abs(frame-frame_base)*mask
    diff = abs(frame-frame_base)
            
    ret,fgmask0 = cv2.threshold(diff,80,255,cv2.THRESH_BINARY)
    fgmask0 = fgmask0*mask
    fgmask1 = cv2.morphologyEx(fgmask0,cv2.MORPH_OPEN,kernel)

    #cv2.imshow('substraction0',fgmask1)
    #k = cv2.waitKey(80) & 0xff
    #if k == 27:
    #    break

    #find contours
    #_,contours,_ = cv2.findContours(fgmask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, hier0 = cv2.findContours(fgmask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
    if len(contours) > 0:

        areas = [] #list to hold all areas
        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)

        max_area = max(areas)
        max_area_index = areas.index(max_area) #index of the list element with largest area

        cnt = contours[max_area_index] #largest area contour

        if len(cnt) > 5:

            cnt_area = cv2.contourArea(cnt)
            cnt_hull_area = cv2.contourArea(cv2.convexHull(cnt))
            cnt_solidity = float(cnt_area)/cnt_hull_area
            #m = cv2.moments(cnt)
            #Centroid = ( m['m10']/m['m00'],m['m01']/m['m00'] )

            # fit ellipse
            (Cx,Cy),(Minor,Major),Angle = cv2.fitEllipse(cnt)
            Eccentricity = np.sqrt(1-(Minor/Major)**2)

            #print ii,length, cnt_area, Cx, Cy, cnt_solidity, Eccentricity, Angle
            csvWriter.writerow((videoBaseName, ii, cnt_area, Cx, Cy, cnt_solidity, Eccentricity, Angle))

            iCenter = (int(Cx), int(Cy))
            AngleMajor = (Angle-90)/180*np.pi
            AngleMinor = (Angle-180)/180*np.pi
            drMajor = ( int(Major/2*np.cos(AngleMajor)), int(Major/2*np.sin(AngleMajor)))
            drMinor = ( int(Minor/2*np.cos(AngleMinor)), int(Minor/2*np.sin(AngleMinor)))

            ellipse = cv2.fitEllipse(cnt)
            #cv2.drawContours(frame0, [cnt], 0, (0, 255, 0), 3, maxLevel = 0)
            cv2.ellipse(frame0,ellipse,(0,0,255),2)
            cv2.circle(frame0,iCenter,5,[0,0,255],-1)
            cv2.putText(frame0,str(int(Angle)), (iCenter[0]+5,iCenter[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)
            cv2.line(frame0, iCenter, (iCenter[0]+drMajor[0], iCenter[1]+drMajor[1]), (0, 0, 255))
            cv2.line(frame0, iCenter, (iCenter[0]+drMinor[0], iCenter[1]+drMinor[1]), (255, 0, 0))

    #cv2.imshow('frame '+str(ii), frame0)
    videoOutWriter.write(frame0)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
toc = time.time()
print toc-tic    
    
cap.release()
videoOutWriter.release()
cv2.destroyAllWindows()
sys.exit(1)

# #cap = cv2.VideoCapture('vtest.avi')
# #cap = cv2.VideoCapture('MP_2014-12-19_10-47-29_600.mov')
# cap = cv2.VideoCapture(fn)
#
# #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# #fgbg = cv2.createBackgroundSubtractorMOG2()
# #fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
# fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
#
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#
# while(1):
#     ret, frame = cap.read()
#     #print frame
#     #fgmask = fgbg.apply(frame)
#     #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
#
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
