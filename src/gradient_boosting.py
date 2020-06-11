import sys
from imutils import paths
import numpy as np
import argparse
import pickle
import time
import cv2
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve

ap = argparse.ArgumentParser()

ap.add_argument("--data", default="data_insight.pickle",
    help='Path to data')
ap.add_argument("--folder", default="matching_out_2865_max/")
ap.add_argument("--models_out", default="gradient_boost.pickle")
args = ap.parse_args()

data = pickle.loads(open(args.folder + args.data, "rb").read())
# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["labels"])
print("Encoder: ", labels)

X = np.array(data['data'])
y = labels
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=123)
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
i = 1
for train_idx, valid_idx in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]

    gbc = GradientBoostingClassifier(random_state=123)
    gbc.fit(X_train, y_train)

    y_pred = gbc.predict(X_test)
    with open(args.folder + 'fold_' + str(i) + '_' + args.models_out, 'wb') as f:
        pickle.dump(gbc, f)
    i += 1
    from sklearn.metrics import classification_report, confusion_matrix
    report = confusion_matrix(y_test,y_pred)
    print(report)
    print(classification_report(y_test,y_pred))

    TP = report[0][0]
    FP = report[1][0]
    TN = report[1][1]
    FN = report[0][1]
    Precision = round(TP/(TP+FP),4)
    Recall = round(TP/(TP+FN),4)
    Accuracy = round((TP+TN)/(TP+TN+FP+FN),4)
    F_core = round(2*Precision*Recall/(Precision+Recall),4)
    print("[INFO] Precision = {}".format(Precision))
    print("[INFO] Accuracy = {}".format(Accuracy))
    print("[INFO] Recall = {}".format(Recall))
    print("[INFO] F_core = {}".format(F_core))
'''
#ROC
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred)

# Plot
plt.figure(1)
plt.plot(fpr_rt_lm, tpr_rt_lm)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
#plt.legend(loc='best')

plt.legend(['ROC AUC'], loc='lower right')
plt.savefig('gb.png')
'''
