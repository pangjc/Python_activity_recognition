# version 0 works for the MIT experimental set up

print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt
import pandas as pd

featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2_features.csv'
#featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_20150514CagesTwoMice_features.csv'

activities = ['0','1-3','4-6']
df = pd.read_csv(featureFileFullName,header = False)

df.columns = ['iframe','Hu1[0][0]','Hu1[1][0]','Hu1[2][0]','Hu1[3][0]','Hu1[4][0]','Hu1[5][0]','Hu1[6][0]',
               'Hu2[0][0]','Hu2[1][0]','Hu2[2][0]','Hu2[3][0]','Hu2[4][0]','Hu2[5][0]','Hu2[6][0]',
                'cx1', 'cy1', 'cx2', 'cy2', 'meiSize', 'corner1', 'corner2', 'corner3', 'corner4',
                'height', 'width', 'extend', 'JD_score','SMO_score']
df['JD_score_group'] = 0
df['JD_score_group'][(1<=df['JD_score'])&(df['JD_score']<=3)] = 1
df['JD_score_group'][(4<=df['JD_score'])&(df['JD_score']<=6)] = 2


df['SMO_score_group'] = 0
df['SMO_score_group'][(1<=df['SMO_score'])&(df['SMO_score']<=3)] = 1
df['SMO_score_group'][(4<=df['SMO_score'])&(df['SMO_score']<=6)] = 2
nrow,ncol = df.shape
# Create a classifier: a support vector classifier
print 'Loading SVM ...'
from sklearn.externals import joblib
classifier = joblib.load('mySVM_JAK1.pkl')

#classifier.fit(my_data_train, my_label_train)
# Now predict the value of the digit on the second half:
expected_train = df['JD_score_group'].values
trainColumns = df.columns[1:ncol-4]
print trainColumns
my_data_train = df[trainColumns].values
predicted_train = classifier.predict(my_data_train)


print("Classification report for training set %s:\n%s\n"
      % (classifier, metrics.classification_report(expected_train, predicted_train)))
print("Confusion matrix for training set:\n%s" % metrics.confusion_matrix(expected_train, predicted_train))

cm = metrics.confusion_matrix(expected_train, predicted_train)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.matshow(cm_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize=20)
plt.colorbar()

activitiesNames = ['No Seizures','Seizure 1-3','Seizure 4-6']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize=15)
plt.yticks(tick_marks, activitiesNames,fontsize=15)
    
plt.ylabel('JD Score',fontsize=18)
plt.xlabel('Predicted label',fontsize=18)
plt.show()

from scipy.signal import medfilt
predicted_train = medfilt(predicted_train, 13)

print("Classification report for training set %s:\n%s\n"
      % (classifier, metrics.classification_report(expected_train, predicted_train)))
print("Confusion matrix for training set:\n%s" % metrics.confusion_matrix(expected_train, predicted_train))

cm = metrics.confusion_matrix(expected_train, predicted_train)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.matshow(cm_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize=20)
plt.colorbar()

activitiesNames = ['No Seizures','Seizure 1-3','Seizure 4-6']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize=15)
plt.yticks(tick_marks, activitiesNames,fontsize=15)
    
plt.ylabel('JD Score',fontsize=18)
plt.xlabel('Predicted label',fontsize=18)
plt.show()

df['predict'] = predicted_train
featureFileFullNameSave = featureFileFullName[0:-4]+ '_predicted.csv'

ts = np.linspace(0,(len(predicted_train)-1)/900.0,len(predicted_train))
plt.plot(ts,predicted_train,'r*',df['JD_score_group'],'b*',df['SMO_score_group'],'y*')
plt.show()
df.to_csv(featureFileFullNameSave)