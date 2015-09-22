# version 0 works for the MIT experimental set up
# Switch JD and SMO's annotation
print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt

#featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2_features.csv'
featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_20150514CagesTwoMice_features.csv'

activities = ['0','1-3','4-6']

cvfolder = 2
subFactor = 1
my_data_label = genfromtxt(featureFileFullName, delimiter=',')
(nrow, ncol) = my_data_label.shape 

my_data_label0 = my_data_label[my_data_label[:,ncol-1]==0]
my_data_label0[:,ncol-1] = 0
num_data0 = my_data_label0.shape[0] 
num_data0_sub = num_data0/subFactor
trainInd0=np.random.randint(num_data0,size=num_data0_sub)
my_data_label0_train = my_data_label0[trainInd0[0:num_data0_sub/cvfolder],:]
my_data_label0_test = my_data_label0[trainInd0[num_data0_sub/cvfolder:num_data0_sub],:]

my_data_label13 = my_data_label[(my_data_label[:,ncol-1]>=1)&(my_data_label[:,ncol-1]<=3)]
my_data_label13[:,ncol-1] = 1
num_data13 = my_data_label13.shape[0] 
num_data13_sub = num_data13/subFactor
trainInd13=np.random.randint(num_data13,size=num_data13_sub)
my_data_label13_train = my_data_label13[trainInd13[0:num_data13_sub/cvfolder],:]
my_data_label13_test = my_data_label13[trainInd13[num_data13_sub/cvfolder:num_data13_sub],:]

my_data_label46 = my_data_label[(my_data_label[:,ncol-1]>=4)&(my_data_label[:,ncol-1]<=6)]
my_data_label46[:,ncol-1] = 2
num_data46 = my_data_label46.shape[0] 
num_data46_sub = num_data46/subFactor
trainInd46=np.random.randint(num_data46,size=num_data46_sub)
my_data_label46_train = my_data_label46[trainInd46[0:num_data46_sub/cvfolder],:]
my_data_label46_test = my_data_label46[trainInd46[num_data46_sub/cvfolder:num_data46_sub],:]


###### Added in 8/18/2015 for quantitative validation
######
my_data_label0_manual = my_data_label[my_data_label[:,ncol-1]==0]
my_data_label0_valid = my_data_label0_manual[trainInd0[num_data0_sub/cvfolder:num_data0_sub],:]

my_data_label13_manual = my_data_label[(my_data_label[:,ncol-1]>=1)&(my_data_label[:,ncol-1]<=3)]
my_data_label13_valid = my_data_label13_manual[trainInd13[num_data13_sub/cvfolder:num_data13_sub],:]

my_data_label46_manual = my_data_label[(my_data_label[:,ncol-1]>=4)&(my_data_label[:,ncol-1]<=6)]
my_data_label46_valid = my_data_label46_manual[trainInd46[num_data46_sub/cvfolder:num_data46_sub],:]

my_data_label_valid = np.concatenate((my_data_label0_valid,my_data_label13_valid),axis=0) 
my_data_label_valid = np.concatenate((my_data_label_valid,my_data_label46_valid),axis=0) 

num_col_valid = len(my_data_label_valid[0])
my_label_valid0 = my_data_label_valid[:,num_col_valid-2]
my_label_valid0 = np.asarray(my_label_valid0,dtype=np.int)

my_label_valid = my_label_valid0
my_label_valid[(my_label_valid0>=1)&(my_label_valid0<=3)] = 1
my_label_valid[(my_label_valid0>=4)&(my_label_valid0<=6)] = 2
######
######
num_col = len(my_data_label[0])


my_data_label_train = np.concatenate((my_data_label0_train,my_data_label13_train),axis=0) 
my_data_label_train = np.concatenate((my_data_label_train,my_data_label46_train),axis=0) 

my_data_label_test = np.concatenate((my_data_label0_test,my_data_label13_test),axis=0) 
my_data_label_test = np.concatenate((my_data_label_test,my_data_label46_test),axis=0) 

print 'training data size: ' + str(my_data_label_train.shape)
print 'test data size: ' + str(my_data_label_test.shape)

my_data_train = my_data_label_train[:,1:num_col-2]  
#featureInd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,26]
#my_data_train = my_data_label_train[:,featureInd] 
my_label_train = my_data_label_train[:,num_col-1]
my_label_train = my_label_train.astype(int)

my_data_test = my_data_label_test[:,1:num_col-2]  
#my_data_test = my_data_label_test[:,featureInd] 
my_label_test = my_data_label_test[:,num_col-1]
my_label_test = my_label_test.astype(int)
# Create a classifier: a support vector classifier
print 'start SVM training ...'
classifier = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  gamma=0.00001, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)

classifier.fit(my_data_train, my_label_train)
print 'SVM training finished!'
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
joblib.dump(classifier,'mySVM.pkl')

# Draw confusion matrix 
cm = metrics.confusion_matrix(expected_test, predicted_test)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.matshow(cm_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize = 20)
plt.colorbar()

activitiesNames = ['No Seizures','Seizure 1-3','Seizure 4-6']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize = 15)
plt.yticks(tick_marks, activitiesNames,fontsize = 15)
    
plt.ylabel('SMO Score',fontsize = 18)
plt.xlabel('Predicted label',fontsize = 18)
plt.show()

# classifier1 = joblib.load('mySVM.pkl')
## Two rater evaluates between each other
rater_JD = my_data_label[:,ncol-2]
rater_SMO = my_data_label[:,ncol-1]

rater_JD_group = rater_JD
rater_JD_group[(rater_JD>=1)&(rater_JD<=3)] = 1
rater_JD_group[(rater_JD>=4)&(rater_JD<=6)] = 2

rater_SMO_group = rater_SMO
rater_SMO_group[(rater_SMO>=1)&(rater_SMO<=3)] = 1
rater_SMO_group[(rater_SMO>=4)&(rater_SMO<=6)] = 2


# Draw confusion matrix 
cm2 = metrics.confusion_matrix(rater_SMO_group, rater_JD_group)
cm2_normalized = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
plt.matshow(cm2_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize = 20)
plt.colorbar()

activitiesNames = ['No Seizures','Seizure 1-3','Seizure 4-6']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize = 15)
plt.yticks(tick_marks, activitiesNames,fontsize = 15)
    
plt.ylabel('SMO Score',fontsize = 18)
plt.xlabel('JD Score',fontsize = 18)
plt.show()

print('Manual evaluation: \n%s' % metrics.classification_report(rater_SMO_group,rater_JD_group))


# Draw confusion matrix 
cm3 = metrics.confusion_matrix(predicted_test, my_label_valid)
cm3_normalized = cm3.astype('float') / cm3.sum(axis=1)[:, np.newaxis]
plt.matshow(cm3_normalized, vmin=0, vmax=1.0)
plt.title('Normalized confusion matrix',fontsize = 20)
plt.colorbar()

activitiesNames = ['No Seizures','Seizure 1-3','Seizure 4-6']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames,fontsize=15)
plt.yticks(tick_marks, activitiesNames,fontsize=15)
    
plt.ylabel('Predicted label',fontsize = 18)
plt.xlabel('JD Score',fontsize = 18)
plt.show()

print('Manual evaluation: \n%s' % metrics.classification_report(predicted_test, my_label_valid))