# version 2
# instead using grouping, use stages 0, 1, 2, 3, 4, 5 and 6
print(__doc__)

import csv
import numpy as np
from numpy import genfromtxt
from sklearn import svm, metrics
import pickle
import matplotlib.pyplot as plt

#featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_Cam2_20150514Cages1_2_features.csv'
featureFileFullName = 'C:\\PostDoctorProjects\\VideoEEGData\\20150514_JD_Cam2_Cages1_2_combined\\JD_20150514CagesTwoMice_features.csv'

activities = ['0','1','2','3','4','5','6']

cvfolder = 2
subFactor = 1
my_data_label = genfromtxt(featureFileFullName, delimiter=',')
(nrow, ncol) = my_data_label.shape 

my_data_label0 = my_data_label[my_data_label[:,ncol-2]==0]
num_data0 = my_data_label0.shape[0] 
num_data0_sub = num_data0/subFactor
trainInd=np.random.randint(num_data0,size=num_data0_sub)
my_data_label0_train = my_data_label0[trainInd[0:num_data0_sub/cvfolder],:]
my_data_label0_test = my_data_label0[trainInd[num_data0_sub/cvfolder:num_data0_sub],:]

my_data_label1 = my_data_label[my_data_label[:,ncol-2]==1]
num_data1 = my_data_label1.shape[0] 
num_data1_sub = num_data1/subFactor
trainInd=np.random.randint(num_data1,size=num_data1_sub)
my_data_label1_train = my_data_label1[trainInd[0:num_data1_sub/cvfolder],:]
my_data_label1_test = my_data_label1[trainInd[num_data1_sub/cvfolder:num_data1_sub],:]

my_data_label2 = my_data_label[my_data_label[:,ncol-2]==2]
num_data2 = my_data_label2.shape[0] 
num_data2_sub = num_data2/subFactor
trainInd=np.random.randint(num_data2,size=num_data2_sub)
my_data_label2_train = my_data_label2[trainInd[0:num_data2_sub/cvfolder],:]
my_data_label2_test = my_data_label2[trainInd[num_data2_sub/cvfolder:num_data2_sub],:]

my_data_label3 = my_data_label[my_data_label[:,ncol-2]==3]
num_data3 = my_data_label3.shape[0] 
num_data3_sub = num_data3/subFactor
trainInd=np.random.randint(num_data3,size=num_data3_sub)
my_data_label3_train = my_data_label3[trainInd[0:num_data3_sub/cvfolder],:]
my_data_label3_test = my_data_label3[trainInd[num_data3_sub/cvfolder:num_data3_sub],:]

my_data_label4 = my_data_label[my_data_label[:,ncol-2]==4]
num_data4 = my_data_label4.shape[0] 
num_data4_sub = num_data4/subFactor
trainInd=np.random.randint(num_data4,size=num_data4_sub)
my_data_label4_train = my_data_label4[trainInd[0:num_data4_sub/cvfolder],:]
my_data_label4_test = my_data_label4[trainInd[num_data4_sub/cvfolder:num_data4_sub],:]

my_data_label5 = my_data_label[my_data_label[:,ncol-2]==5]
num_data5 = my_data_label5.shape[0] 
num_data5_sub = num_data5/subFactor
trainInd=np.random.randint(num_data5,size=num_data5_sub)
my_data_label5_train = my_data_label5[trainInd[0:num_data5_sub/cvfolder],:]
my_data_label5_test = my_data_label5[trainInd[num_data5_sub/cvfolder:num_data5_sub],:]

my_data_label6 = my_data_label[my_data_label[:,ncol-2]==6]
num_data6 = my_data_label6.shape[0] 
num_data6_sub = num_data6/subFactor
trainInd=np.random.randint(num_data6,size=num_data6_sub)
my_data_label6_train = my_data_label6[trainInd[0:num_data6_sub/cvfolder],:]
my_data_label6_test = my_data_label6[trainInd[num_data6_sub/cvfolder:num_data6_sub],:]

my_data_label_train = np.concatenate((my_data_label0_train,my_data_label1_train),axis=0) 
my_data_label_train = np.concatenate((my_data_label_train,my_data_label2_train),axis=0) 
my_data_label_train = np.concatenate((my_data_label_train,my_data_label3_train),axis=0) 
my_data_label_train = np.concatenate((my_data_label_train,my_data_label4_train),axis=0) 
my_data_label_train = np.concatenate((my_data_label_train,my_data_label5_train),axis=0) 
my_data_label_train = np.concatenate((my_data_label_train,my_data_label6_train),axis=0) 


my_data_label_test = np.concatenate((my_data_label0_test,my_data_label1_test),axis=0) 
my_data_label_test = np.concatenate((my_data_label_test,my_data_label2_test),axis=0) 
my_data_label_test = np.concatenate((my_data_label_test,my_data_label3_test),axis=0) 
my_data_label_test = np.concatenate((my_data_label_test,my_data_label4_test),axis=0) 
my_data_label_test = np.concatenate((my_data_label_test,my_data_label5_test),axis=0) 
my_data_label_test = np.concatenate((my_data_label_test,my_data_label6_test),axis=0) 

num_col = len(my_data_label[0])

print 'training data size: ' + str(my_data_label_train.shape)
print 'test data size: ' + str(my_data_label_test.shape)

featureInd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
#my_data_train = my_data_label_train[:,1:num_col-2]  
my_data_train = my_data_label_train[:,featureInd]  
my_label_train = my_data_label_train[:,num_col-2]

## Shuffle data to double check
#num_data = my_data_train.shape[0]
#trainInd=np.random.randint(num_data,size=num_data)
#my_label_train = my_label_train[trainInd]
##


my_label_train = my_label_train.astype(int)

#my_data_test = my_data_label_test[:,1:num_col-2]  
my_data_test = my_data_label_test[:,featureInd]  
my_label_test = my_data_label_test[:,num_col-2]
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
plt.title('Normalized confusion matrix')
plt.colorbar()

activitiesNames = ['0','1','2','3','4','5','6']
tick_marks = np.arange(len(activitiesNames))
plt.xticks(tick_marks, activitiesNames)
plt.yticks(tick_marks, activitiesNames)
    
plt.ylabel('JD Score')
plt.xlabel('Predicted label')
plt.show()

# classifier1 = joblib.load('mySVM.pkl')

