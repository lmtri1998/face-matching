from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K

class SoftMax():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        print("Y_true: ", y_true)
        print("Y_pred: ", y_pred)
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    def contrastive_loss1(self, y_true, y_pred):
        margin = 1
        return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    def build(self):
        model = Sequential()
        model.add(Dense(2048, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(2048, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))
        rms = RMSprop()
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005, amsgrad=False)
        model.compile(loss=self.contrastive_loss1,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model
