import sys
from imutils import paths
import pandas as pd
import numpy as np
import argparse
import pickle
import time
import cv2
import os
from softmax import SoftMax
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.svm import SVC

ap = argparse.ArgumentParser()

ap.add_argument("--data", default="data_insight.pickle",
    help='Path to data')
ap.add_argument("--folder", default="matching_out_2865_random/")
ap.add_argument("--type", default="RANDOM")
args = ap.parse_args()

out_roc = args.folder + "ROC_{}.png".format(args.type)

data = pickle.loads(open(args.folder + args.data, "rb").read())
# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["labels"])
num_classes = len(np.unique(labels))

#--------------------------
labels__ = labels.reshape(-1, 1)
one_hot_encoder = OneHotEncoder(categorical_features = [0])
labels__ = one_hot_encoder.fit_transform(labels__).toarray()
#--------------------------
print("Encoder: ", labels)

X = np.array(data['data'])
y = labels
y__ = labels__ # Softmax

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=123)

#X_train, X_test, y_train__, y_test__ = train_test_split(X,y__, test_size = 0.20, random_state=123) # softmax


model = []

input_shape = X.shape[1]

# Append model Deep Learning
for i in range(1):
    print(args.folder + "best_fold_{}.hdf5".format(str(i+1)))
    model.append(load_model(args.folder + "best_fold_{}.hdf5".format(str(i+1))))

# Append model Machine Learning
models_ml = ['svm', 'random_forest', 'gradient_boost', 'voting']
for ml in models_ml:
    for i in range(1):
        model.append(pickle.loads(open(args.folder + 'fold_{}_{}.pickle'.format(str(i+1), ml), "rb").read()))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score

# Create KFold
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
# Train

def convert_label(y, FAR=0):
    if FAR != 0:
        y = np.where(y > 1 - FAR, 1, 0)
    y_preds = []
    for i in range(len(y)):
        if y[i][0] == 1:
            y_preds.append(0)
        else:
            y_preds.append(1)
    return y_preds

label = ["SoftMax","SVM", "RF", "GB", "VT"]
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

i = 0
X_test = None
y_test = None
for train_idx, valid_idx in cv.split(X):
    if i >= 1 : break
    print("-------------------------------------------")
    X_train, X_val, y_train, y_val = X[train_idx], X[valid_idx], y__[train_idx], y__[valid_idx]

    X_test = X_val
    y_test = y[valid_idx]

    y_preds = model[i].predict_proba(X_val)[:,1]
    y_val_ = convert_label(y_val)
    fpr, tpr, _ = roc_curve(y_val_,  y_preds)
    i += 1
    auc = roc_auc_score(y_val_, y_preds)
    result_table = result_table.append({'classifiers':"SoftMax",
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)
for i in range(4):
    y_pred = model[i+1].predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    auc = roc_auc_score(y_test, y_pred)
    result_table = result_table.append({'classifiers':label[i+1],
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)

fig = plt.figure(figsize=(8,6))
plt.grid()
for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(label[i], result_table.loc[i]['auc']))

plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Accept Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Accept Rate", fontsize=15)

plt.title('{} COSINE SIMILARITY - ROC Curve Analysis'.format(args.type), fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.savefig(out_roc)