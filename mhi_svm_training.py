print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle

featureFolderPath = 'C:\\InternProjects\\rat_activity_recognition\\MIT_Traning_samples\\_Feature_Exact\\'

activities = ['drink','eat','groom','hang','head','rear','rest','walk']

my_data_label0 = genfromtxt(featureFolderPath + activities[0] + '_features.csv', delimiter=',')
my_data_label1 = genfromtxt(featureFolderPath + activities[1] + '_features.csv', delimiter=',')
my_data_label2 = genfromtxt(featureFolderPath + activities[2] + '_features.csv', delimiter=',')
my_data_label3 = genfromtxt(featureFolderPath + activities[3] + '_features.csv', delimiter=',')
my_data_label4 = genfromtxt(featureFolderPath + activities[4] + '_features.csv', delimiter=',')
my_data_label5 = genfromtxt(featureFolderPath + activities[5] + '_features.csv', delimiter=',')
my_data_label6 = genfromtxt(featureFolderPath + activities[6] + '_features.csv', delimiter=',')
my_data_label7 = genfromtxt(featureFolderPath + activities[7] + '_features.csv', delimiter=',')

my_data_label = np.concatenate((my_data_label0,my_data_label3),axis=0)
my_data_label = np.concatenate((my_data_label, my_data_label2),axis=0)
my_data_label = np.concatenate((my_data_label, my_data_label3),axis=0)
my_data_label = np.concatenate((my_data_label, my_data_label4),axis=0)
my_data_label = np.concatenate((my_data_label, my_data_label5),axis=0)
my_data_label = np.concatenate((my_data_label, my_data_label6),axis=0)
my_data_label = np.concatenate((my_data_label, my_data_label7),axis=0)

print my_data_label.shape
my_data = my_data_label[:,15:18]  
my_label = my_data_label[:,17]
my_label = my_label.astype(int)

# Create a classifier: a support vector classifier
classifier = svm.SVC()

# We learn the digits on the first half of the digits

classifier.fit(my_data, my_label)
svm.SVC()

# Now predict the value of the digit on the second half:
expected = my_label
predicted = classifier.predict(my_data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# write the SVM to a file
outputSVM = open('mySVM.pkl', 'wb')
pickle.dump(classifier, outputSVM)
outputSVM.close()

# read trained SVM back from the file
trainedSVM_pkl = open('mySVM.pkl', 'rb')
classifier_load = pickle.load(trainedSVM_pkl)
trainedSVM_pkl.close()

predicted = classifier_load.predict(my_data)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))