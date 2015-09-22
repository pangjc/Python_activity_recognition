import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time
import scipy.io as io

videoName = 'C:\PostDoctorProjects\VideoEEGData\overnight831_901\Sync_stitched\Sync_2015-08-31_16-40-12_600.avi'

cam = cv2.VideoCapture(videoName)
video_len = cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
ret, frame0 = cam.read()
hei, wid = frame0.shape[:2]

frame = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

print wid, hei, video_len

frame = cv2.GaussianBlur(frame,(3,3),0)  

roi10 = frame[3:30,220:245]


roi20 = frame[100:180,230:300]

'''
plt.subplot(211)
plt.imshow(roi10)
plt.subplot(212)
plt.imshow(roi20)
plt.show()
'''        
roi10 = cv2.Canny(roi10,15,20)
roi20 = cv2.Canny(roi20,15,20)

kk = 0
diffInt1 = np.zeros(video_len-1)
diffInt2 = np.zeros(video_len-1)

tic = time.time()
for iFrame in xrange(int(video_len)-1):

    ret, frame1 = cam.read()
    frame1 = cv2.GaussianBlur(frame1,(3,3),0) 
    roi11 = frame1[3:30,220:245]
    roi11 = cv2.Canny(roi11,15,20)

    roi21 = frame1[100:180,230:300]
    if (kk%1000 == 0): 
        print str(kk) + " out of " + str(video_len)
        cv2.imshow('led',roi21)
        if 0xff & cv2.waitKey(1) == 27:
            break
    roi21 = cv2.Canny(roi21,15,20)
    
    diff1 = np.matrix(np.absolute(np.subtract(roi11,roi10)))
    diffInt1[kk] = diff1.sum()   
    
    diff2 = np.matrix(np.absolute(np.subtract(roi21,roi20)))
    diffInt2[kk] = diff2.sum() 
    
    roi10 = roi11
    roi20 = roi21
    
    kk = kk + 1 

    
cam.release()
toc = time.time()

print "total time: " + str(toc-tic)

SynVideo_LED = {}
SynVideo_LED['videoTimeStamp'] = diffInt1
SynVideo_LED['LED'] = diffInt2
SynVideo_LED['Name'] = videoName
io.savemat('C:\PostDoctorProjects\VideoEEGData\overnight831_901\Sync_stitched\SynVideo_LED0831',SynVideo_LED)

plt.subplot(211)
plt.plot(diffInt1[0:kk-1])
plt.subplot(212)
plt.plot(diffInt2[0:kk-1])
plt.show()

