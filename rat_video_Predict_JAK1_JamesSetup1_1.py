# version 0 works for the MIT experimental set up
#
# Manual scoring removed
print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt
import pandas as pd

featureFileFullName = 'C:\\PostDoctorProjects\VideoEEGData\\20150520_Cam4_Subject_5+6_stitched\\20150520_Cam4_Subject_5+6_08-20-04_stitched_features.csv'
#featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_20150514CagesTwoMice_features.csv'

activities = ['0','1-3','4-6']
df = pd.read_csv(featureFileFullName,header = False)

df.columns = ['iframe','Hu1[0][0]','Hu1[1][0]','Hu1[2][0]','Hu1[3][0]','Hu1[4][0]','Hu1[5][0]','Hu1[6][0]',
               'Hu2[0][0]','Hu2[1][0]','Hu2[2][0]','Hu2[3][0]','Hu2[4][0]','Hu2[5][0]','Hu2[6][0]',
                'cx1', 'cy1', 'cx2', 'cy2', 'meiSize', 'corner1', 'corner2', 'corner3', 'corner4',
                'height', 'width', 'extend']
nrow,ncol = df.shape
# Create a classifier: a support vector classifier
print 'Loading SVM ...'
from sklearn.externals import joblib
classifier = joblib.load('mySVM_JAK1.pkl')

#classifier.fit(my_data_train, my_label_train)
# Now predict the value of the digit on the second half:

trainColumns = df.columns[1:ncol]
my_data_train = df[trainColumns].values
predicted_train = classifier.predict(my_data_train)


from scipy.signal import medfilt
predicted_train = medfilt(predicted_train, 13)


df['predict'] = predicted_train
featureFileFullNameSave = featureFileFullName[0:-4]+ '_predicted.csv'

ts = np.linspace(0,(len(predicted_train)-1)/900.0,len(predicted_train))
plt.plot(ts,predicted_train,'r')
plt.show()
df.to_csv(featureFileFullNameSave, index = False)