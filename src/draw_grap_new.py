import sys
from imutils import paths
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
ap.add_argument("--type", default="MAX")
args = ap.parse_args()

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
for i in range(5):
    print(args.folder + "best_fold_{}.hdf5".format(str(i+1)))
    model.append(load_model(args.folder + "best_fold_{}.hdf5".format(str(i+1))))
    print(model)

# Append model Machine Learning
models_ml = ['svm', 'random_forest', 'gradient_boost', 'voting']
for ml in models_ml:
    for i in range(5):
        model.append(pickle.loads(open(args.folder + 'fold_{}_{}.pickle'.format(str(i+1), ml), "rb").read()))

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

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

i = 0
acc_sm = 0
for train_idx, valid_idx in cv.split(X):
    #print(valid_idx)
    X_train, X_val, y_train, y_val = X[train_idx], X[valid_idx], y__[train_idx], y__[valid_idx]
    y_pred = model[i].predict(X_val)
    i += 1
    y_preds = convert_label(y_pred, 0.5)
    y_val_ = convert_label(y_val)

    report = confusion_matrix(y_val_,y_preds)
    #print(report)
    #print(classification_report(y_val_,y_preds))

    TP = report[0][0]
    FP = report[1][0]
    TN = report[1][1]
    FN = report[0][1]

    #Precision = round(TP/(TP+FP),4)
    #Recall = round(TP/(TP+FN),4)
    Accuracy = round((TP+TN)/(TP+TN+FP+FN),4)

    acc_sm += Accuracy

    #F_core = round(2*Precision*Recall/(Precision+Recall),4)
    #print("[INFO] Precision = {}".format(Precision))
    print("[INFO] Accuracy {} = {}".format(Accuracy, i))
    #print("[INFO] Recall = {}".format(Recall))
    #print("[INFO] F_core = {}".format(F_core))

acc_ML = []
# Create KFold
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

for train_idx, valid_idx in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]
    for j in range(4):
        y_pred = model[i+j*5].predict(X_test)
        report = confusion_matrix(y_test,y_pred)
        #print(report)
        #print(classification_report(y_test,y_pred))

        TP = report[0][0]
        FP = report[1][0]
        TN = report[1][1]
        FN = report[0][1]

        #Precision = round(TP/(TP+FP),4)
        #Recall = round(TP/(TP+FN),4)
        Accuracy = round((TP+TN)/(TP+TN+FP+FN),4)
        acc_ML.append(Accuracy)
        #F_core = round(2*Precision*Recall/(Precision+Recall),4)
        #print("[INFO] Precision = {}".format(Precision))
        #print("[INFO] Accuracy = {}".format(Accuracy))
        #print("[INFO] Recall = {}".format(Recall))
        #print("[INFO] F_core = {}".format(F_core))
    i += 1
    print("----------------------------------")

accuracy = []
acc_svm = 0
acc_rf = 0
acc_gb = 0
acc_vt = 0
for i in range(20):
    if i%4 == 0:
        acc_svm += acc_ML[i]
    elif i%4 == 1:
        acc_rf += acc_ML[i]
        print(acc_ML[i])
    elif i%4 == 2:
        acc_gb += acc_ML[i]
    elif i%4 == 3:
        acc_vt += acc_ML[i]

accuracy.append(round((1 - acc_svm/5)*100, 4)) # SVM
accuracy.append(round((1 - acc_rf/5)*100, 4)) # RF
accuracy.append(round((1 - acc_gb/5)*100, 4)) # GB
accuracy.append(round((1 - acc_vt/5)*100, 4)) # VT
accuracy.append(round((1 - acc_sm/5)*100, 4)) # Softmax

label = ["SVM", "RF", "GB", "VT", "SoftMax"]

import matplotlib.cm as cm
from matplotlib.colors import Normalize
# Get a color map
my_cmap = cm.get_cmap('jet')

# Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
my_norm = Normalize(vmin=0, vmax=5)

x = np.arange(len(label))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x, accuracy, width, color=my_cmap(my_norm(accuracy)))
rects1 = ax.bar(x, accuracy, width, color=['C0', 'C1', 'C2', 'C3', 'C4'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error rate (%)')
ax.set_title('{} COSINE SIMILARITY'.format(args.type))
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend()
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
fig.tight_layout()
plt.savefig(args.folder + "bar_chart_{}.png".format(args.type))
