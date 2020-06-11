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
#ap.add_argument("--models_out", default="svm.pickle")
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=123)

X_train, X_test, y_train__, y_test__ = train_test_split(X,y__, test_size = 0.20, random_state=123) # softmax


model = []

input_shape = X.shape[1]

model_svm = pickle.loads(open(args.folder + "svm.pickle", "rb").read())
model_rf = pickle.loads(open(args.folder + "rf.pickle", "rb").read())
model_gb = pickle.loads(open(args.folder + "gb.pickle", "rb").read())
model_vt = pickle.loads(open(args.folder + "vt.pickle", "rb").read())

softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
models = softmax.build()

model_s1 = load_model(args.folder + "best_fold_1.hdf5")
model_s2 = load_model(args.folder + "best_fold_2.hdf5")
model_s3 = load_model(args.folder + "best_fold_3.hdf5")
model_s4 = load_model(args.folder + "best_fold_4.hdf5")
model_s5 = load_model(args.folder + "best_fold_5.hdf5")

model.append(model_s1)
model.append(model_s2)
model.append(model_s3)
model.append(model_s4)
model.append(model_s5)

model.append(model_svm)
model.append(model_rf)
model.append(model_gb)
model.append(model_vt)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# Create KFold
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
# Train
i = 0
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

acc = 0
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

    Precision = round(TP/(TP+FP),4)
    Recall = round(TP/(TP+FN),4)
    Accuracy = round((TP+TN)/(TP+TN+FP+FN),4)

    acc += Accuracy

    F_core = round(2*Precision*Recall/(Precision+Recall),4)
    #print("[INFO] Precision = {}".format(Precision))
    print("[INFO] Accuracy {} = {}".format(Accuracy, i))
    #print("[INFO] Recall = {}".format(Recall))
    #print("[INFO] F_core = {}".format(F_core))

accuracy = []

for i in range(4):
    y_pred = model[i+5].predict(X_test)

    report = confusion_matrix(y_test,y_pred)
    #print(report)
    #print(classification_report(y_test,y_pred))

    TP = report[0][0]
    FP = report[1][0]
    TN = report[1][1]
    FN = report[0][1]

    Precision = round(TP/(TP+FP),4)
    Recall = round(TP/(TP+FN),4)
    Accuracy = round((TP+TN)/(TP+TN+FP+FN),4)
    accuracy.append(round((1 - Accuracy)*100,4))
    F_core = round(2*Precision*Recall/(Precision+Recall),4)
    #print("[INFO] Precision = {}".format(Precision))
    #print("[INFO] Accuracy = {}".format(Accuracy))
    #print("[INFO] Recall = {}".format(Recall))
    #print("[INFO] F_core = {}".format(F_core))

acc = round(acc,4)

accuracy.append(round((1 - acc/5)*100, 4))

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
ax.set_title('RANDOM COSINE SIMILARITY')
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
#plt.savefig(args.folder + "bar_chart.png")
