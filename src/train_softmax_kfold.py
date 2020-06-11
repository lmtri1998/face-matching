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
ap.add_argument("--folder", default="./matching_out_2865_random/")

args = vars(ap.parse_args())

data = pickle.loads(open(args["folder"] + args["data"], "rb").read())

#import numpy, scipy.io
#scipy.io.savemat('./data_2865_max.mat', mdict={'data': data})


# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["labels"])
num_classes = len(np.unique(labels))
labels = labels.reshape(-1, 1)
one_hot_encoder = OneHotEncoder(categorical_features = [0])
labels = one_hot_encoder.fit_transform(labels).toarray()
print("Encoder: ", labels)
#labels= np.where(labels == 0, 1, 0)
#print("Encoder: ", labels)

y = labels
X = np.array(data['data'])
# Initialize Softmax training model arguments
BATCH_SIZE = 32
EPOCHS = 20
input_shape = X.shape[1]
# Build sofmax classifier
softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
model = softmax.build()

csv_logger = CSVLogger(args['folder'] + "training.log", append=True)
model_checkpoint_path = args['folder'] + "weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
model_checkpointer = ModelCheckpoint(model_checkpoint_path, save_best_only=False, save_weights_only=False, monitor="val_acc", period=1, mode='max')
#tensorboard = TensorBoard(log_dir=job_dir + '/logs/', histogram_freq=0, write_graph=True, write_images=True)
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
        math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
callbacks = [csv_logger, model_checkpointer]


# Create KFold
cv = KFold(n_splits = 5, random_state=123, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
# Train
for train_idx, valid_idx in cv.split(X):
    print(valid_idx)
    softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
    model = softmax.build()
    X_train, X_val, y_train, y_val = X[train_idx], X[valid_idx], y[train_idx], y[valid_idx]
    his = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(X_val, y_val), callbacks=callbacks)
    print(his.history['acc'])

    history['acc'] += his.history['acc']
    history['val_acc'] += his.history['val_acc']
    history['loss'] += his.history['loss']
    history['val_loss'] += his.history['val_loss']
'''
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

    plt.figure(figsize=(15, 9))
    plt.plot(b,a)
    #plt.axis([0,1,0,1])
    plt.legend(['Softmax ({})'.format(tpr)])
    plt.xlabel('FAR')
    plt.ylabel('VR')
    plt.show()
    plt.savefig(args["folder"] + 'ROC.png')
ROC(y_val, y_pred)
# write the face recognition model to output
#model.save(args['models_out'])

# Plot
#plt.figure(1, figsize=(8, 12))
# Summary history for accuracy
#plt.subplot(211)
#plt.plot(history['acc'])
#plt.plot(history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')

# Summary history for loss
#plt.subplot(212)
#plt.plot(history['loss'])
#plt.plot(history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig(args["folder"] + 'accuracy_loss.png')
#plt.show()
'''
