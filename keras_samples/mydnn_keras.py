from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
 
def MyDNN(input_shape=(32, 32, 1), output_size=10, learning_rate=0.001, keep_prob=0.5):
    model = Sequential()
    
    model.add(Conv2D(20, kernel_size=5, strides=2, activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())
 
    model.add(Conv2D(50, kernel_size=5, strides=2, activation='relu'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())
 
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(output_size, activation='softmax'))
 
    model.compile(optimizer=Adam(lr=learning_rate, epsilon=1e-1),
                  loss='categorical_crossentropy', metrics=['accuracy'])

