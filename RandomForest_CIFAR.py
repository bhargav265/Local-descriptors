from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("LBP_features_train.csv",header = None)
target = pd.read_csv("train_labels.csv",header = None)

train = train.values

train = train.astype(np.int)
#train =  np.multiply(train,1.0/1000.0)
target = target.values.ravel()
target = target.astype(np.int)


test = pd.read_csv("LBP_features_test.csv", header = None).values

test = test.astype(np.int)

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
for h in range(10):
	est = [10,50,100,200,400,800,1000,1500,2000,2500]
	rf = RandomForestClassifier(n_estimators=est[h], n_jobs=-1)
	rf.fit(train, target)
	pred = rf.predict(test)

	actual_pred = pd.read_csv("test_labels.csv", header = None).values.ravel()

	actual_pred = actual_pred.astype(np.int)

	count = 0;

	for i in range(len(pred)):
		if(pred[i]==actual_pred[i]):
			count += 1

	print('Accuracy = ' + str(count/10000) + ' for no. of trees = ' + str(est[h]))


