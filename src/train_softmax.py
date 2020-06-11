from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import load_model
#from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
from softmax import SoftMax
import numpy as np
import argparse
import pickle

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
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.20, random_state=123)
print(X.shape)
# Initialize Softmax training model arguments
BATCH_SIZE = 32
EPOCHS = 5
input_shape = X.shape[1]
print(input_shape)
# Build sofmax classifier
softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
model = softmax.build()

# Create KFold
cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
# Train
his = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(X_val, y_val))
print(his.history['acc'])

history['acc'] += his.history['acc']
history['val_acc'] += his.history['val_acc']
history['loss'] += his.history['loss']
history['val_loss'] += his.history['val_loss']

# write the face recognition model to output
model.save(args['model'])

# Plot
plt.figure(1)
# Summary history for accuracy
plt.subplot(211)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# Summary history for loss
plt.subplot(212)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(args["folder"] + 'accuracy_loss.png')
plt.show()
