from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import LearningRateScheduler
from softmax import SoftMax
import numpy as np
import math
import argparse
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--data", default="data_insight.pickle",
                help="path to data")
ap.add_argument("--models_out", default="model_softmax.h5",
                help="path to output trained model")
ap.add_argument("--folder", default="./matching_out_1250/")

args = vars(ap.parse_args())

data = pickle.loads(open(args["folder"] + args["data"], "rb").read())

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["labels"])
num_classes = len(np.unique(labels))
print(num_classes)
labels = labels.reshape(-1, 1)
one_hot_encoder = OneHotEncoder(categorical_features = [0])
labels = one_hot_encoder.fit_transform(labels).toarray()
print("Encoder: ", labels)
y = labels
X = np.array(data['data'])
input_shape = X.shape[1]
print(X.shape)
# Initialize Softmax training model arguments
softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
model = softmax.build()

model.load_weights(args["folder"] + 'my_model.hdf5')

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.20, random_state=123)
y_pred=model.predict_proba(X_val)
from sklearn.metrics import roc_curve,roc_auc_score

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
def ROC(y_val_, y_pred_):
    a = []
    b = []
    print(y_val_, y_pred_)
    for i in np.arange(0.00001, 0.01, 0.00001):
        if i > 0.50005:
            break
        y_vals = convert_label(y_val_)
        y_preds = convert_label(y_pred_, round(i,5))
        report = confusion_matrix(y_vals,y_preds)
        TP = report[0][0]
        FP = report[1][0]
        TN = report[1][1]
        FN = report[0][1]
        tpr = round(TP/(TP+FN),4)
        #print(tpr)
        a.append(tpr)
        b.append(i)

    import seaborn as sns
    sns.set('talk', 'whitegrid', 'dark', font_scale=1.5, font='Ricty',
         rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

    lw = 2
    plt.figure(figsize=(15, 9))
    #plt.plot(b, a, color='darkorange',
    #     lw=lw)
    plt.semilogx(b,a)
    #plt.xticks(np.arange(4), [1e-5, 1e-4, 1e-3, 1e-2]) 
    #plt.yticks(np.arange(10),[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) 
    #plt.xlim([0.00001, 0.01])
    #plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(args["folder"] + 'ROC.png')
    plt.close()

'''
    plt.figure(figsize=(15, 9))
    plt.plot(b,a)
    #plt.axis([0,1,0,1])
    plt.legend(['Softmax ({})'.format(tpr)])
    plt.xlabel('FAR')
    plt.ylabel('VR')
    plt.show()
    plt.savefig(args["folder"] + 'ROC.png')'''
ROC(y_val, y_pred)
import seaborn as sns
