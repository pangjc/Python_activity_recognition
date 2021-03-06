# version 0 works for the MIT experimental set up
# Modified based on rat_video_Training_JAK1_JamesSetup1.py by using new features consisting of both dynamic and stationary features 
# Created in 9/17/2015
# Created in 9/18/2015 by using two class rather than three classes

print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt
import time
#featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2_features.csv'
featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_20150514CagesTwoMice_Fullfeatures.csv'

activities = ['0','1-3','4-6']

cvfolder = 2
subFactor = 1
my_data_label = genfromtxt(featureFileFullName, delimiter=',')
(nrow, ncol) = my_data_label.shape 


my_data_label03 = my_data_label[(my_data_label[:,ncol-2]>=0)&(my_data_label[:,ncol-2]<=3)]
my_data_label03[:,ncol-2] = 1
num_data03 = my_data_label03.shape[0] 
num_data03_sub = num_data03/subFactor
trainInd03=np.random.randint(num_data03,size=num_data03_sub)
my_data_label03_train = my_data_label03[trainInd03[0:num_data03_sub/cvfolder],:]
my_data_label03_test = my_data_label03[trainInd03[num_data03_sub/cvfolder:num_data03_sub],:]

my_data_label46 = my_data_label[(my_data_label[:,ncol-2]>=4)&(my_data_label[:,ncol-2]<=6)]
my_data_label46[:,ncol-2] = 2
num_data46 = my_data_label46.shape[0] 
num_data46_sub = num_data46/subFactor
trainInd46=np.random.randint(num_data46,size=num_data46_sub)
my_data_label46_train = my_data_label46[trainInd46[0:num_data46_sub/cvfolder],:]
my_data_label46_test = my_data_label46[trainInd46[num_data46_sub/cvfolder:num_data46_sub],:]


###### Added in 8/18/2015 for quantitative validation
######
print('Debugging')

my_data_label03_manual = my_data_label[(my_data_label[:,ncol-2]>=0)&(my_data_label[:,ncol-2]<=3)]
my_data_label03_valid = my_data_label03_manual[trainInd03[num_data03_sub/cvfolder:num_data03_sub],:]

my_data_label46_manual = my_data_label[(my_data_label[:,ncol-2]>=4)&(my_data_label[:,ncol-2]<=6)]
my_data_label46_valid = my_data_label46_manual[trainInd46[num_data46_sub/cvfolder:num_data46_sub],:]

my_data_label_valid = np.concatenate((my_data_label03_valid,my_data_label46_valid),axis=0) 

num_col_valid = len(my_data_label_valid[0])
my_label_valid0 = my_data_label_valid[:,num_col_valid-1]
my_label_valid0 = np.asarray(my_label_valid0,dtype=np.int)

my_label_valid = my_label_valid0
my_label_valid[(my_label_valid0>=0)&(my_label_valid0<=3)] = 1
my_label_valid[(my_label_valid0>=4)&(my_label_valid0<=6)] = 2
######
######
num_col = len(my_data_label[0])


my_data_label_train = np.concatenate((my_data_label03_train,my_data_label46_train),axis=0) 

my_data_label_test = np.concatenate((my_data_label03_test,my_data_label46_test),axis=0) 

print 'training data size: ' + str(my_data_label_train.shape)
print 'test data size: ' + str(my_data_label_test.shape)

#my_data_train = my_data_label_train[:,1:num_col-2]  
#featureInd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,24,25,26]
#featureInd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,24,25,26,27,28,29,30,31,32,33,34,35,36]
featureInd = [8,9,10,11,12,13,14,16,24,25,26,29,30,31,32,33,34,35,36]
my_data_train = my_data_label_train[:,featureInd] 
my_label_train = my_data_label_train[:,num_col-2]
my_label_train = my_label_train.astype(int)

my_data_test = my_data_label_test[:,1:num_col-2]  
my_data_test = my_data_label_test[:,featureInd] 
my_label_test = my_data_label_test[:,num_col-2]
my_label_test = my_label_test.astype(int)
# Create a classifier: a support vector classifier
print 'start SVM training ...'

classifier = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  gamma=0.00001, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)
print 'SVM training finished!'
tic = time.time()
classifier.fit(my_data_train, my_label_train)
toc = time.time()
# Now predict the value of the digit on the second half:
expected_train = my_label_train
predicted_train = classifier.predict(my_data_train)

print("Classification report for training set %s:\n%s\n"
      % (classifier, metrics.classification_report(expected_train, predicted_train)))
print("Confusion matrix for training set:\n%s" % metrics.confusion_matrix(expected_train, predicted_train))

# Now predict the value of the digit on the second half:
expected_test = my_label_test
predicted_test = classifier.predict(my_data_test)

print("Classification report for test set %s:\n%s\n"
      % (classifier, metrics.classification_report(expected_test, predicted_test)))
print("Confusion matrix for test set:\n%s" % metrics.confusion_matrix(expected_test, predicted_test))

# Save the trained classifier into disk
from sklearn.externals import joblib
joblib.dump(classifier,'mySVM_JAK1_0917.pkl')

# Draw confusion matrix 
cm = metrics.confusion_matrix(expected_test, predicted_test)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.matshow(cm_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize=20)
plt.colorbar()

activitiesNames = ['No convulsive Seizures',' convulsive Seizures']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize=15)
plt.yticks(tick_marks, activitiesNames,fontsize=15)
    
plt.ylabel('JD Score',fontsize=18)
plt.xlabel('Predicted label',fontsize=18)
plt.show()

# classifier1 = joblib.load('mySVM.pkl')
## Two rater evaluates between each other
rater_JD = my_data_label[:,ncol-2]
rater_SMO = my_data_label[:,ncol-1]

rater_JD_group = rater_JD
rater_JD_group[(rater_JD>=0)&(rater_JD<=3)] = 1
rater_JD_group[(rater_JD>=4)&(rater_JD<=6)] = 2

rater_SMO_group = rater_SMO
rater_SMO_group[(rater_SMO>=0)&(rater_SMO<=3)] = 1
rater_SMO_group[(rater_SMO>=4)&(rater_SMO<=6)] = 2


# Draw confusion matrix 
cm2 = metrics.confusion_matrix(rater_JD_group, rater_SMO_group)
cm2_normalized = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
plt.matshow(cm2_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize=20)
plt.colorbar()

activitiesNames = ['No convulsive Seizures',' convulsive Seizures']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize=15)
plt.yticks(tick_marks, activitiesNames,fontsize=15)
    
plt.ylabel('JD Score',fontsize=18)
plt.xlabel('SMO Score',fontsize=18)
plt.show()

print('Manual evaluation: \n%s' % metrics.classification_report(rater_JD_group,rater_SMO_group))


# Draw confusion matrix 
cm3 = metrics.confusion_matrix(predicted_test, my_label_valid)
cm3_normalized = cm3.astype('float') / cm3.sum(axis=1)[:, np.newaxis]
plt.matshow(cm3_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize=20)
plt.colorbar()

activitiesNames = ['No convulsive Seizures',' convulsive Seizures']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize=15)
plt.yticks(tick_marks, activitiesNames,fontsize=15)
    
plt.ylabel('Predicted label',fontsize=18)
plt.xlabel('SMO Score',fontsize=18)
plt.show()

print('Manual evaluation: \n%s' % metrics.classification_report(predicted_test, my_label_valid))

print 'Fitting time: ' + str(toc-tic)