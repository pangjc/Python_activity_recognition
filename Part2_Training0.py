# version 0 works for the MIT experimental set up

print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt

featureFolderPath = 'C:\\InternProjects\\rat_activity_recognition\\MIT_Traning_samples\\_Feature_Exact\\'

activities = ['drink','eat','groom','hang','head','rear','rest','walk']

cvfolder = 2
subFactor = 1
my_data_label = genfromtxt(featureFolderPath + activities[0] + '_features.csv', delimiter=',')
num_data = len(my_data_label) 
num_data_sub = num_data/subFactor
trainInd=np.random.randint(num_data,size=num_data_sub)
my_data_label_train = my_data_label[trainInd[0:num_data_sub/cvfolder],:]
my_data_label_test = my_data_label[trainInd[num_data_sub/cvfolder:num_data_sub],:]

num_col = len(my_data_label[0])

for ii in range(1,8):
    my_data_label1 = genfromtxt(featureFolderPath + activities[ii] + '_features.csv', delimiter=',')
    num_data1 = len(my_data_label1) 
    num_data_sub1 = num_data1/subFactor
    trainInd1=np.random.randint(num_data1,size=num_data_sub1)
    my_data_label_train1 = my_data_label1[trainInd1[0:num_data_sub1/cvfolder],:]
    my_data_label_test1 = my_data_label1[trainInd1[num_data_sub1/cvfolder:num_data_sub1],:]
    my_data_label_train = np.concatenate((my_data_label_train,my_data_label_train1),axis=0) 
    my_data_label_test = np.concatenate((my_data_label_test,my_data_label_test1),axis=0) 

print 'training data size: ' + str(my_data_label_train.shape)
print 'test data size: ' + str(my_data_label_test.shape)

my_data_train = my_data_label_train[:,0:num_col-1]  
my_label_train = my_data_label_train[:,num_col-1]
my_label_train = my_label_train.astype(int)

my_data_test = my_data_label_test[:,0:num_col-1]  
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
plt.matshow(cm_normalized)
plt.title('Normalized confusion matrix')
plt.colorbar()

activitiesNames = ['drink','eat','groom','hang','micromovement','rear','rest','walk']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames)
plt.yticks(tick_marks, activitiesNames)
    
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# classifier1 = joblib.load('mySVM.pkl')

