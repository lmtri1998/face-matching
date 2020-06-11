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
import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score

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
#y__ = labels__ # Softmax

model = []

models_ml = ['svm']
for ml in models_ml:
    for i in range(5):
        model.append(pickle.loads(open(args.folder + 'fold_{}_{}.pickle'.format(str(i+1), ml), "rb").read()))

# plot arrows
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')
ax1.add_patch(
    patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)
    )
ax1.add_patch(
    patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)
    )

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 0

# Create KFold
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

for train_idx, valid_idx in cv.split(X):
    print("-------------------------------------------")
    X_train, X_val, y_train, y_val = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]
    y_preds = model[i].predict_proba(X_val)[:,1]
    fpr, tpr, t = roc_curve(y_val, y_preds)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.savefig("test_fold.png")