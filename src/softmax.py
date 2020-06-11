from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras as k

class SoftMax():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def build(self):
        model = Sequential()
        model.add(Dense(2048, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
#        model.add(Dense(2048, activation='relu'))
#        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))
#        rms = RMSprop()
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005, amsgrad=False)
        model.compile(loss=k.losses.binary_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        '''model.compile(loss=k.losses.binary_crossentropy,
                      optimizer=rms,
                      metrics=['accuracy'])'''

        return model
