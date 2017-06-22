#from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("LBP_features_train.csv", header = None)
target = pd.read_csv("train_labels.csv", header = None)

train = train.values

train = train.astype(np.int)
#train =  np.multiply(train,1.0/1000.0)
target = target.values.ravel()
target = target.astype(np.int)


test = pd.read_csv("LBP_features_test.csv").values

test = test.astype(np.int)

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
for j in range(14):
	neigh = [1,3,5,7,9,11,13,15,17,19,21,23,25,27]
	rf = KNeighborsClassifier(n_neighbors=neigh[j],n_jobs = 2)
	rf.fit(train, target)
	pred = rf.predict(test)

	actual_pred = pd.read_csv("test_labels.csv").values.ravel()

	actual_pred = actual_pred.astype(np.int)
    
	count = 0;

	for i in range(len(pred)):
		if(pred[i]==actual_pred[i]):
			count += 1

	print('Accuracy = ' + str(count/10000) + ' for Neighbors = ' + str(neigh[j]))


