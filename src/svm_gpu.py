import sys

from imutils import paths
import numpy as np
import argparse
import pickle
import time
#import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.svm import SVC
from cuml.svm import SVC

ap = argparse.ArgumentParser()

ap.add_argument("--data", default="data.pickle",
    help='Path to data')
ap.add_argument("--folder", default="matching_out/")
ap.add_argument("--models_out", default="svm_gpu.pickle")
args = ap.parse_args()

data = pickle.loads(open(args.folder + args.data, "rb").read())
# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["labels"])
print("Encoder: ", labels)

X = np.array(data['data'])

y = labels
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)

svclassifier = SVC()

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(y_pred)
print(type(y_pred))
#with open(args.folder + args.models_out, 'wb') as f:
#    pickle.dump(svclassifier, f)
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
