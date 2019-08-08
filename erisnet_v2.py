from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, GlobalAveragePooling1D, LSTM, Dropout

from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras import regularizers


input_dim = 14

def createModel_ERISNet():
   """Create the ERISNet neural network."""
   model = Sequential()
   model.add(Reshape(input_shape=(input_dim, 1), target_shape=(1, input_dim)))
   for i in range(3):
       model.add(Conv1D(64, 6, padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
       model.add(BatchNormalization())
       model.add(Dropout(0.1))

   for i in range(3):
       model.add(Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
       model.add(BatchNormalization())
       model.add(Dropout(0.1))

   for i in range(3):
       model.add(Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
       model.add(BatchNormalization())
       model.add(Dropout(0.1))

   for i in range(2):
       model.add(LSTM(units=64, return_sequences=True, activation='tanh'))
       model.add(BatchNormalization())

   model.add(Dropout(0.5))
   model.add(GlobalAveragePooling1D())

   model.add(Dense(2, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
   model.summary()

   return model