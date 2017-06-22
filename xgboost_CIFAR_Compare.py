import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve,roc_curve,auc,accuracy_score,average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
# create the training & test sets, skipping the header row with [1:]
train_LDMP = pd.read_csv("CSLDMPfeatures_train.csv",header = None)
train_LBP = pd.read_csv("CSLDPfeatures_train.csv",header = None)
train_LDP = pd.read_csv("CSLBPfeatures_train.csv",header = None)
train_LMP = pd.read_csv("CSLMPfeatures_train.csv",header = None)
train_XCSLBP = pd.read_csv("XCSLBPfeatures_train.csv",header = None)
train_XCSLMP = pd.read_csv("XCSLMPfeatures_train.csv",header = None)

target = pd.read_csv("Labels/train_labels.csv",header = None)
colors1 = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
colors2 = cycle([ 'turquoise', 'darkorange', 'cornflowerblue', 'teal','navy'])
train_LDMP = train_LDMP.values
train_LBP = train_LBP.values
train_LDP = train_LDP.values
train_LMP = train_LMP.values
train_XCSLBP = train_XCSLBP.values
train_XCSLMP = train_XCSLMP.values

train_LDMP = train_LDMP.astype(np.int)
train_LBP = train_LBP.astype(np.int)
train_LDP = train_LDP.astype(np.int)
train_LMP = train_LMP.astype(np.int)
train_XCSLBP = train_XCSLBP.astype(np.int)
train_XCSLMP = train_XCSLMP.astype(np.int)
#train =  np.multiply(train,1.0/1000.0)
target = target.values.ravel()
target = target.astype(np.int)


test_LDMP = pd.read_csv("CSLDMPfeatures_test.csv", header = None).values
test_LBP = pd.read_csv("CSLDPfeatures_test.csv", header = None).values
test_LDP = pd.read_csv("CSLBPfeatures_test.csv", header = None).values
test_LMP = pd.read_csv("CSLMPfeatures_test.csv", header = None).values
test_XCSLBP = pd.read_csv("XCSLBPfeatures_test.csv", header = None).values
test_XCSLMP = pd.read_csv("XCSLMPfeatures_test.csv", header = None).values

test_LDMP = test_LDMP.astype(np.int)
test_LBP = test_LBP.astype(np.int)
test_LDP = test_LDP.astype(np.int)
test_LMP = test_LMP.astype(np.int)
test_XCSLBP = test_XCSLBP.astype(np.int)
test_XCSLMP = test_XCSLMP.astype(np.int)

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
for h in range(2):
	est = [1000,1500]
	rf_LBP = XGBClassifier(n_estimators=est[h])
	rf_LDMP = XGBClassifier(n_estimators=est[h])
	rf_LDP = XGBClassifier(n_estimators=est[h])
	rf_LMP = XGBClassifier(n_estimators=est[h])
	rf_XCSLBP = XGBClassifier(n_estimators=est[h])
	rf_XCSLMP = XGBClassifier(n_estimators=est[h])

	score_LBP = rf_LBP.fit(train_LBP, target).predict_proba(test_LBP)
	score_LDMP = rf_LDMP.fit(train_LDMP, target).predict_proba(test_LDMP)
	score_LDP = rf_LDP.fit(train_LDP, target).predict_proba(test_LDP)
	score_LMP = rf_LMP.fit(train_LMP, target).predict_proba(test_LMP)
	score_XCSLBP = rf_XCSLBP.fit(train_XCSLBP, target).predict_proba(test_XCSLBP)
	score_XCSLMP = rf_XCSLMP.fit(train_XCSLMP, target).predict_proba(test_XCSLMP)

	pred_LBP = rf_LBP.predict(test_LBP)
	pred_LDMP = rf_LDMP.predict(test_LDMP)
	pred_LDP = rf_LDP.predict(test_LDP)
	pred_LMP = rf_LMP.predict(test_LMP)
	pred_XCSLBP = rf_XCSLBP.predict(test_XCSLBP)
	pred_XCSLMP = rf_XCSLMP.predict(test_XCSLMP)
	#pred = rf.predict(test)

	actual_pred = pd.read_csv("Labels/test_labels.csv", header = None).values.ravel()

	actual_pred = actual_pred.astype(np.int)
	actual_pred_nb = actual_pred

	actual_pred = label_binarize(actual_pred, classes=[0,1,2,3,4,5,6,7,8,9])
	
	precision_LBP = dict()
	recall_LBP = dict()
	pr_auc_LBP = dict()
	precision_LDMP = dict()
	recall_LDMP = dict()
	pr_auc_LDMP = dict()
	precision_LDP = dict()
	recall_LDP = dict()
	pr_auc_LDP = dict()
	precision_LMP = dict()
	recall_LMP = dict()
	pr_auc_LMP = dict()
	precision_XCSLBP = dict()
	recall_XCSLBP = dict()
	pr_auc_XCSLBP = dict()
	precision_XCSLMP = dict()
	recall_XCSLMP = dict()
	pr_auc_XCSLMP = dict()


	fpr_LBP = dict()
	tpr_LBP = dict()
	roc_auc_LBP = dict()
	fpr_LDMP = dict()
	tpr_LDMP = dict()
	roc_auc_LDMP = dict()
	fpr_LDP = dict()
	tpr_LDP = dict()
	roc_auc_LDP = dict()
	fpr_LDP = dict()
	tpr_LDP = dict()
	roc_auc_LDP = dict()
	fpr_LMP = dict()
	tpr_LMP = dict()
	roc_auc_LMP = dict()
	fpr_XCSLBP = dict()
	tpr_XCSLBP = dict()
	roc_auc_XCSLBP = dict()
	fpr_XCSLMP = dict()
	tpr_XCSLMP = dict()
	roc_auc_XCSLMP = dict()





	precision_LBP["micro"], recall_LBP["micro"], _ = precision_recall_curve(actual_pred.ravel(),score_LBP.ravel())
	pr_auc_LBP["micro"] = average_precision_score(actual_pred, score_LBP,
                                                     average="micro")
	precision_LDMP["micro"], recall_LDMP["micro"], _ = precision_recall_curve(actual_pred.ravel(),score_LDMP.ravel())
	pr_auc_LDMP["micro"] = average_precision_score(actual_pred, score_LDMP,
                                                     average="micro")
	precision_LDP["micro"], recall_LDP["micro"], _ = precision_recall_curve(actual_pred.ravel(),score_LDP.ravel())
	pr_auc_LDP["micro"] = average_precision_score(actual_pred, score_LDP,
                                                     average="micro")
	precision_LMP["micro"], recall_LMP["micro"], _ = precision_recall_curve(actual_pred.ravel(),score_LMP.ravel())
	pr_auc_LMP["micro"] = average_precision_score(actual_pred, score_LMP,
                                                     average="micro")
	precision_XCSLBP["micro"], recall_XCSLBP["micro"], _ = precision_recall_curve(actual_pred.ravel(),score_XCSLBP.ravel())
	pr_auc_XCSLBP["micro"] = average_precision_score(actual_pred, score_XCSLBP,
                                                     average="micro")
	precision_XCSLMP["micro"], recall_XCSLMP["micro"], _ = precision_recall_curve(actual_pred.ravel(),score_XCSLMP.ravel())
	pr_auc_XCSLMP["micro"] = average_precision_score(actual_pred, score_XCSLMP,
                                                     average="micro")

	fpr_LBP["micro"], tpr_LBP["micro"], _ = roc_curve(actual_pred.ravel(), score_LBP.ravel())
	roc_auc_LBP["micro"] = auc(fpr_LBP["micro"], tpr_LBP["micro"])
	fpr_LDMP["micro"], tpr_LDMP["micro"], _ = roc_curve(actual_pred.ravel(), score_LDMP.ravel())
	roc_auc_LDMP["micro"] = auc(fpr_LDMP["micro"], tpr_LDMP["micro"])
	fpr_LDP["micro"], tpr_LDP["micro"], _ = roc_curve(actual_pred.ravel(), score_LDP.ravel())
	roc_auc_LDP["micro"] = auc(fpr_LDP["micro"], tpr_LDP["micro"])
	fpr_LMP["micro"], tpr_LMP["micro"], _ = roc_curve(actual_pred.ravel(), score_LMP.ravel())
	roc_auc_LMP["micro"] = auc(fpr_LMP["micro"], tpr_LMP["micro"])
	fpr_XCSLBP["micro"], tpr_XCSLBP["micro"], _ = roc_curve(actual_pred.ravel(), score_XCSLBP.ravel())
	roc_auc_XCSLBP["micro"] = auc(fpr_XCSLBP["micro"], tpr_XCSLBP["micro"])
	fpr_XCSLMP["micro"], tpr_XCSLMP["micro"], _ = roc_curve(actual_pred.ravel(), score_XCSLMP.ravel())
	roc_auc_XCSLMP["micro"] = auc(fpr_XCSLMP["micro"], tpr_XCSLMP["micro"])


	plt.plot(recall_LBP["micro"], precision_LBP["micro"], color='red', lw=2,
            label='CSLDP, auc = {0}'.format(pr_auc_LBP["micro"]))
	plt.plot(recall_LDMP["micro"], precision_LDMP["micro"], color='blue', lw=2,
             label='CSLDMP, auc = {0}'.format(pr_auc_LDMP["micro"]))
	plt.plot(recall_LDP["micro"], precision_LDP["micro"], color='green', lw=2,
             label='CSLBP, auc = {0}'.format(pr_auc_LDP["micro"]))
	plt.plot(recall_LMP["micro"], precision_LMP["micro"], color='orange', lw=2,
             label='CSLMP, auc = {0}'.format(pr_auc_LMP["micro"]))
	plt.plot(recall_XCSLBP["micro"], precision_XCSLBP["micro"], color='pink', lw=2,
            label='XCSLBP, auc = {0}'.format(pr_auc_XCSLBP["micro"]))
	plt.plot(recall_XCSLMP["micro"], precision_XCSLMP["micro"], color='black', lw=2,
            label='XCSLMP, auc = {0}'.format(pr_auc_XCSLMP["micro"]))


	plt.xlabel('Recall')
	plt.ylabel('Precision')	
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.legend(loc="upper right")
	plt.title('Precision Recall Curve ' + str(est[h]))
	plt.show()


	plt.plot(fpr_LBP["micro"], tpr_LBP["micro"], color='red',
         lw=2, label='CSLDP, auc = {0}'.format(roc_auc_LBP["micro"]))
	plt.plot(fpr_LDMP["micro"], tpr_LDMP["micro"], color='blue',
         lw=2, label='CSLDMP, auc = {0}'.format(roc_auc_LDMP["micro"]))
	plt.plot(fpr_LDP["micro"], tpr_LDP["micro"], color='green',
         lw=2, label='CSLBP, auc = {0}'.format(roc_auc_LDP["micro"]))
	plt.plot(fpr_LMP["micro"], tpr_LMP["micro"], color='orange',
         lw=2, label='CSLMP, auc = {0}'.format(roc_auc_LMP["micro"]))
	plt.plot(fpr_XCSLBP["micro"], tpr_XCSLBP["micro"], color='pink',
         lw=2, label='XCSLBP, auc = {0}'.format(roc_auc_XCSLBP["micro"]))
	plt.plot(fpr_XCSLMP["micro"], tpr_XCSLMP["micro"], color='black',
         lw=2, label='XCSLMP, auc = {0}'.format(roc_auc_XCSLMP["micro"]))

	plt.plot([0, 1], [0, 1], color='gold', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic '+ str(est[h]))
	plt.legend(loc="lower right")
	plt.show()

	print("accuracy for CSLDP "+str(est[h]) +" est is "+str(accuracy_score(actual_pred_nb,pred_LBP)))
	print("accuracy for CSLDMP "+str(est[h]) +" est is "+str(accuracy_score(actual_pred_nb,pred_LDMP)))
	print("accuracy for CSLBP "+str(est[h]) +" est is "+str(accuracy_score(actual_pred_nb,pred_LDP)))
	print("accuracy for CSLMP "+str(est[h]) +" est is "+str(accuracy_score(actual_pred_nb,pred_LMP)))
	print("accuracy for XCSLBP "+str(est[h]) +" est is "+str(accuracy_score(actual_pred_nb,pred_XCSLBP)))
	print("accuracy for XCSLMP "+str(est[h]) +" est is "+str(accuracy_score(actual_pred_nb,pred_XCSLMP)))
	print(" ")


	
	

	
